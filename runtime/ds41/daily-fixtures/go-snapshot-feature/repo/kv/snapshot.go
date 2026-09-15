package kv
// Snapshot returns a stable portable encoding of the current map.
func (s *Store) Snapshot() ([]byte,error) { panic("TODO") }
// Restore atomically replaces the map only when the entire encoding is valid.
func (s *Store) Restore(data []byte) error { panic("TODO") }
