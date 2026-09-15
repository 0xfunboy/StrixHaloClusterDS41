package kv
import "sync"
type Store struct{ mu sync.RWMutex; data map[string]string }
func New()*Store{return &Store{data:map[string]string{}}}
func(s *Store)Set(k,v string){s.mu.Lock();defer s.mu.Unlock();s.data[k]=v}
func(s *Store)Get(k string)(string,bool){s.mu.RLock();defer s.mu.RUnlock();v,ok:=s.data[k];return v,ok}
