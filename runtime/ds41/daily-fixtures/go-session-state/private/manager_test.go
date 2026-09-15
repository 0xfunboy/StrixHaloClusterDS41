package session
import "testing"
func TestStaleCompletionCannotCrossGeneration(t *testing.T){var m Manager; first:=m.Start();m.Cancel(first);second:=m.Start();m.Complete(first,"stale");id,r,res:=m.Snapshot();if id!=second||!r||res!=""{t.Fatalf("stale crossed generation id=%d run=%v res=%q",id,r,res)};m.Complete(second,"ok");id,r,res=m.Snapshot();if id!=second||r||res!="ok"{t.Fatalf("valid completion lost")}}
func TestWrongCancelIgnored(t *testing.T){var m Manager; id:=m.Start();m.Cancel(id+99);_,r,_:=m.Snapshot();if !r{t.Fatal("wrong cancel stopped active")}}
