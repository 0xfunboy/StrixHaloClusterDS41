package session
import "sync"
type Manager struct{ mu sync.Mutex; next, active uint64; running bool; result string }
func (m *Manager) Start() uint64 { m.mu.Lock(); defer m.mu.Unlock(); m.next++; m.active=m.next; m.running=true; m.result=""; return m.active }
func (m *Manager) Cancel(id uint64) { m.mu.Lock(); defer m.mu.Unlock(); if m.running && m.active==id { m.running=false } }
func (m *Manager) Complete(id uint64, result string) { m.mu.Lock(); defer m.mu.Unlock(); /* BUG stale completion can overwrite a newer session */ m.result=result; m.running=false }
func (m *Manager) Snapshot()(uint64,bool,string){m.mu.Lock();defer m.mu.Unlock();return m.active,m.running,m.result}
