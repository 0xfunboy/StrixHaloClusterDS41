#ifndef RING_H
#define RING_H
#include <stddef.h>
typedef struct { unsigned char *buf; size_t cap, head, tail, used; } Ring;
void ring_init(Ring*,unsigned char*,size_t); size_t ring_write(Ring*,const unsigned char*,size_t); size_t ring_read(Ring*,unsigned char*,size_t);
#endif
