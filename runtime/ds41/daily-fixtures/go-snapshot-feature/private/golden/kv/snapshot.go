package kv
import("bytes";"encoding/json";"fmt";"io";"sort")
type pair struct{K string `json:"k"`; V string `json:"v"`}
func(s *Store)Snapshot()([]byte,error){s.mu.RLock();defer s.mu.RUnlock();ks:=make([]string,0,len(s.data));for k:=range s.data{ks=append(ks,k)};sort.Strings(ks);p:=make([]pair,0,len(ks));for _,k:=range ks{p=append(p,pair{k,s.data[k]})};return json.Marshal(p)}
func(s *Store)Restore(data []byte)error{dec:=json.NewDecoder(bytes.NewReader(data));dec.DisallowUnknownFields();var p []pair;if err:=dec.Decode(&p);err!=nil{return err};var extra any;if err:=dec.Decode(&extra);err!=io.EOF{return fmt.Errorf("trailing data")};m:=map[string]string{};prev:="";for i,x:=range p{if i>0&&x.K<=prev{return fmt.Errorf("keys not strictly sorted")};if _,ok:=m[x.K];ok{return fmt.Errorf("duplicate key")};m[x.K]=x.V;prev=x.K};s.mu.Lock();s.data=m;s.mu.Unlock();return nil}
