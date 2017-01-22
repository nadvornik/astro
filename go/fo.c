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

int pos = 0;

unsigned int motor_pins[4] = {
	SUNXI_GPA(8),
	SUNXI_GPA(9),
	SUNXI_GPA(10),
	SUNXI_GPA(20)
};

void motor_move(int m)
{
	int step = 1;
        if (m < 0) step = -1;
        m *= step;

        for (int i = 0; i < m; i++) {
		pos += step;
		
		sunxi_gpio_output(motor_pins[pos & 3], 1);
		usleep(1300);
		for (int p = 0; p < 4; p++) {
			if (p != (pos & 3)) sunxi_gpio_output(motor_pins[p], 0);
		}
		usleep(1300);
	}
}

void motor_init()
{
	sunxi_gpio_init();
	for (int i = 0; i < 4; i++) {
		sunxi_gpio_set_cfgpin(motor_pins[i], SUNXI_GPIO_OUTPUT);
	}
	for (int i = 0; i < 4; i++) {
		sunxi_gpio_output(motor_pins[i], 1);
		usleep(10000);
		sunxi_gpio_output(motor_pins[i], 0);
	}
	pos = 0;
}

int main(int argc, char *argv[])
{
	fd_set rfds;

	char buf[BUFSIZE];
	int bufpos = 0;
	
	struct sched_param param;
	
	setpriority(PRIO_PROCESS, 0, 0);
	
	param.sched_priority = 99;
	if (sched_setscheduler(0, SCHED_FIFO, & param) != 0) {
		//printf("sched_setscheduler failed\n");
		//return 1;
	}
	
	motor_init();

	setbuf(stdout, NULL);
	
	while(1) {
		int n;
		
		FD_ZERO(&rfds);
 		FD_SET(0, &rfds);

		n = select(1, &rfds, NULL, NULL, NULL);
		
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
				int m;
				if (sscanf(buf, "%d", &m) != 1) {
					return 1;
				}
				bufpos -= end + 1;
				memmove(buf, endline + 1, bufpos);
				
				motor_move(m);
			
				printf("%d\n", pos);
			}
		}
	}
}
