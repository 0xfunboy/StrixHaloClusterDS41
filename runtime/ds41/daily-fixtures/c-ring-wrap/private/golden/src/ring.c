#include "ring.h"
#include <string.h>
void ring_init(Ring*r,unsigned char*b,size_t c){r->buf=b;r->cap=c;r->head=r->tail=r->used=0;}
size_t ring_write(Ring*r,const unsigned char*src,size_t n){ if(n>r->cap-r->used)n=r->cap-r->used; size_t done=0; while(done<n){ size_t span=r->cap-r->head; if(span>n-done)span=n-done; if(span==0){r->head=0;continue;} memcpy(r->buf+r->head,src+done,span); r->head=(r->head+span)%r->cap; done+=span;} r->used+=done; return done; }
size_t ring_read(Ring*r,unsigned char*dst,size_t n){ if(n>r->used)n=r->used; size_t done=0; while(done<n){size_t span=r->cap-r->tail;if(span>n-done)span=n-done;memcpy(dst+done,r->buf+r->tail,span);r->tail=(r->tail+span)%r->cap;done+=span;}r->used-=done;return done;}
