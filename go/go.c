#include <sys/select.h>

#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>


#include <sched.h>

#include "gpio_lib.h"

#define BUFSIZE 4096

int dec = 0;

unsigned int st4pins[4] = {
	SUNXI_GPC(12),
	SUNXI_GPC(11),
	SUNXI_GPC(10),
	SUNXI_GPC(9)
};

void set_pin(int v)
{
	int p = 3;
	int n = 0;
	if (dec) {
		p = 1;
		n = 2;
	}

	if (v > 0)
                sunxi_gpio_output(st4pins[p], 1);
        else
                sunxi_gpio_output(st4pins[p], 0);
        if (v < 0)
                sunxi_gpio_output(st4pins[n], 1);
        else
                sunxi_gpio_output(st4pins[n], 0);
}

void st4_init()
{
	sunxi_gpio_init();
	if (dec) {
		sunxi_gpio_set_cfgpin(st4pins[1], SUNXI_GPIO_OUTPUT);
		sunxi_gpio_set_cfgpin(st4pins[2], SUNXI_GPIO_OUTPUT);
	}
	else {
		sunxi_gpio_set_cfgpin(st4pins[0], SUNXI_GPIO_OUTPUT);
		sunxi_gpio_set_cfgpin(st4pins[3], SUNXI_GPIO_OUTPUT);
	}
}

int main(int argc, char *argv[])
{
	fd_set rfds;
	struct timeval t_stop;

	char buf[BUFSIZE];
	int bufpos = 0;
	
	struct sched_param param;
	
	int a_len = strlen(argv[0]);
	if (a_len > 3 && strcmp(argv[0] + a_len - 3, "dec") == 0) {
		dec = 1;
	}
	
	setpriority(PRIO_PROCESS, 0, 0);
	
	param.sched_priority = 99;
	if (sched_setscheduler(0, SCHED_FIFO, & param) != 0) {
		//printf("sched_setscheduler failed\n");
		//return 1;
	}
	
	st4_init();

	setbuf(stdout, NULL);
	
	timerclear(&t_stop);

	while(1) {
		struct timeval t_now;
		struct timeval tv;
		int n;
		
		FD_ZERO(&rfds);
 		FD_SET(0, &rfds);

		gettimeofday(&t_now, NULL);

		if (timerisset(&t_stop) && timercmp(&t_stop, &t_now, >)) {
			timersub(&t_stop, &t_now, &tv);
			n = select(1, &rfds, NULL, NULL, &tv);
		}
		else {
			n = select(1, &rfds, NULL, NULL, NULL);
		}
		
		if (n < 0) {
			return 1;
		}
		if (n > 0) {
			char *endline;
			int r = read(0, buf + bufpos, BUFSIZE - bufpos);
			if (r < 0) return 1;
			bufpos += r;
			if (bufpos == BUFSIZE) return 1;
			endline = memchr(buf, '\n', bufpos);
			if (endline) {
				int end = endline - buf;
				buf[end] = 0;
				int v, t;
				if (sscanf(buf, "%d %d", &v, &t) != 2) {
					return 1;
				}
				bufpos -= end + 1;
				memmove(buf, endline + 1, bufpos);
			
				gettimeofday(&t_now, NULL);
				set_pin(v);
				if (t) {
					tv.tv_sec = t / 1000000;
					tv.tv_usec = t % 1000000;
					timeradd(&t_now, &tv, &t_stop);
				}
				else {
					timerclear(&t_stop);
				}
				printf("%d %ld %ld\n", v, t_now.tv_sec,  t_now.tv_usec);
			}
		}
		gettimeofday(&t_now, NULL);
		if (timerisset(&t_stop) && !timercmp(&t_stop, &t_now, >)) {
			set_pin(0);
			timerclear(&t_stop);
			printf("%d %ld %ld\n", 0, t_now.tv_sec,  t_now.tv_usec);
		}
	}
}
