package kv
import("bytes";"testing")
func TestStableRoundTrip(t *testing.T){s:=New();s.Set("z","9");s.Set("a","1");a,e:=s.Snapshot();if e!=nil{t.Fatal(e)};b,e:=s.Snapshot();if e!=nil||!bytes.Equal(a,b){t.Fatal("snapshot nondeterministic")};n:=New();if e=n.Restore(a);e!=nil{t.Fatal(e)};if v,ok:=n.Get("z");!ok||v!="9"{t.Fatal("restore lost z")}}
func TestRestoreAtomicOnInvalid(t *testing.T){s:=New();s.Set("keep","yes");if e:=s.Restore([]byte(`[{"k":"b","v":"2"},{"k":"a","v":"1"}]`));e==nil{t.Fatal("unsorted accepted")};if v,_:=s.Get("keep");v!="yes"{t.Fatal("invalid restore mutated state")};if e:=s.Restore([]byte(`[] {}`));e==nil{t.Fatal("trailing JSON accepted")}}
