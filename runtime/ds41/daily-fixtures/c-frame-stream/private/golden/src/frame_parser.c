#include "frame_parser.h"
#include <string.h>
void frame_parser_init(FrameParser *p){ p->len=0; p->dropping=0; }
int frame_parser_feed(FrameParser *p,const unsigned char *data,size_t n,char out[][128],size_t max_out,size_t *out_count){
  size_t produced=0;
  for(size_t i=0;i<n;i++){
    unsigned char c=data[i];
    if(c=='\n'){
      if(!p->dropping){
        size_t len=p->len; if(len && p->buf[len-1]=='\r') len--;
        if(produced>=max_out) return -2;
        memcpy(out[produced],p->buf,len); out[produced][len]='\0'; produced++;
      }
      p->len=0; p->dropping=0; continue;
    }
    if(p->dropping) continue;
    if(p->len>=127){ p->dropping=1; continue; }
    p->buf[p->len++]=c;
  }
  /* BUG: partial frames are state and must survive feed() chunk boundaries. */
  *out_count=produced; return 0;
}
