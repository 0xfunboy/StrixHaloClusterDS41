#ifndef FRAME_PARSER_H
#define FRAME_PARSER_H
#include <stddef.h>
typedef struct { unsigned char buf[128]; size_t len; int dropping; } FrameParser;
void frame_parser_init(FrameParser *p);
int frame_parser_feed(FrameParser *p, const unsigned char *data, size_t n, char out[][128], size_t max_out, size_t *out_count);
#endif
