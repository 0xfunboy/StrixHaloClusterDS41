package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"time"
)

func (a *App) lifecycleStatus(ctx context.Context) map[string]any {
	if a.cfg.LifecycleCommand == "" {
		return map[string]any{"managed": false, "state": "UNMANAGED"}
	}
	cc, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cc, a.cfg.LifecycleCommand, "status").CombinedOutput()
	var v map[string]any
	if err == nil {
		err = json.Unmarshal(out, &v)
	}
	if v == nil {
		v = map[string]any{}
	}
	v["managed"] = true
	v["preset"] = a.cfg.LifecyclePreset
	a.lifecycleMu.Lock()
	action, last := a.lifecycleAction, a.lifecycleError
	a.lifecycleMu.Unlock()
	if action != "" {
		if action == "on" {
			v["state"] = "STARTING"
		} else {
			v["state"] = "STOPPING"
		}
		v["action"] = action
	}
	if err != nil {
		v["state"] = "ERROR"
		v["error"] = fmt.Sprintf("lifecycle status failed: %v: %.1000s", err, out)
	} else if last != "" {
		v["last_error"] = last
	}
	return v
}

func (a *App) requireModelReady(ctx context.Context) error {
	if a.cfg.LifecycleCommand == "" {
		return nil
	}
	s := a.lifecycleStatus(ctx)
	state, _ := s["state"].(string)
	if state == "READY" {
		return nil
	}
	if state == "RESEARCH_BUSY" {
		return errors.New("DS41 pair is occupied by the research run; K2 serving is unavailable")
	}
	if msg, _ := s["error"].(string); msg != "" {
		return errors.New(msg)
	}
	return fmt.Errorf("DS41 model is %s; explicit authenticated ON is required", state)
}

func (a *App) startLifecycleAction(action string) error {
	if action != "on" && action != "off" {
		return errors.New("unsupported lifecycle action")
	}
	if a.cfg.LifecycleCommand == "" {
		return errors.New("model lifecycle is not configured")
	}
	a.lifecycleMu.Lock()
	if a.lifecycleAction != "" {
		a.lifecycleMu.Unlock()
		return errors.New("lifecycle transition already in progress")
	}
	a.lifecycleAction = action
	a.lifecycleError = ""
	a.lifecycleMu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 35*time.Minute)
		defer cancel()
		out, err := exec.CommandContext(ctx, a.cfg.LifecycleCommand, action).CombinedOutput()
		a.lifecycleMu.Lock()
		a.lifecycleAction = ""
		if err != nil {
			a.lifecycleError = fmt.Sprintf("%s failed: %v: %.2000s", action, err, out)
		}
		a.lifecycleMu.Unlock()
		a.journal.Log("model_lifecycle_"+action, map[string]any{"error": fmt.Sprint(err), "output": string(out)})
	}()
	return nil
}

func (a *App) registerLifecycleRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/lifecycle", func(w http.ResponseWriter, r *http.Request) {
		if !a.authorized(w, r) {
			return
		}
		jsonReply(w, 200, a.lifecycleStatus(r.Context()))
	})
	do := func(action string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if !a.authorized(w, r) {
				return
			}
			var body struct {
				Confirm bool `json:"confirm"`
			}
			if e := decodeBody(w, r, &body); e != nil || !body.Confirm {
				jsonReply(w, 400, map[string]string{"error": "explicit confirm:true required"})
				return
			}
			if e := a.startLifecycleAction(action); e != nil {
				jsonReply(w, 409, map[string]string{"error": e.Error()})
				return
			}
			jsonReply(w, 202, a.lifecycleStatus(r.Context()))
		}
	}
	mux.HandleFunc("POST /v1/lifecycle/on", do("on"))
	mux.HandleFunc("POST /v1/lifecycle/off", do("off"))
}
