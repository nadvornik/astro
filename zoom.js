function isElementInViewport(el) {
    var rect = el.getBoundingClientRect();

    return rect.bottom > 0 &&
        rect.right > 0 &&
        rect.left < (window.innerWidth || document. documentElement.clientWidth) /*or $(window).width() */ &&
        rect.top < (window.innerHeight || document. documentElement.clientHeight) /*or $(window).height() */;
}

var beep_snd = new Audio("data:audio/mp3;base64,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qJrx4KOQjahZxWbcZgztj2c49nKmkId44S71j0c8eV9yDK6uPRzx5X18eDvjvQ6yKo9ZSS6l//8elePK/Lf//IInrOF/FvDoADYAGBMGb7FtErm5MXMlmPAJQVgWta7Zx2go+8xJ0UiCb8LHHdftWyLJE0QIAIsI+UbXu67dZMjmgDGCGl1H+vpF4NSDckSIkk7Vd+sxEhBQMRU8j/12UIRhzSaUdQ+rQU5kGeFxm+hb1oh6pWWmv3uvmReDl0UnvtapVaIzo1jZbf/pD6ElLqSX+rUmOQNpJFa/r+sa4e/pBlAABoAAAAA3CUgShLdGIxsY7AUABPRrgCABdDuQ5GC7DqPQCgbbJUAoRSUj+NIEig0YfyWUho1VBBBA//uQZB4ABZx5zfMakeAAAAmwAAAAF5F3P0w9GtAAACfAAAAAwLhMDmAYWMgVEG1U0FIGCBgXBXAtfMH10000EEEEEECUBYln03TTTdNBDZopopYvrTTdNa325mImNg3TTPV9q3pmY0xoO6bv3r00y+IDGid/9aaaZTGMuj9mpu9Mpio1dXrr5HERTZSmqU36A3CumzN/9Robv/Xx4v9ijkSRSNLQhAWumap82WRSBUqXStV/YcS+XVLnSS+WLDroqArFkMEsAS+eWmrUzrO0oEmE40RlMZ5+ODIkAyKAGUwZ3mVKmcamcJnMW26MRPgUw6j+LkhyHGVGYjSUUKNpuJUQoOIAyDvEyG8S5yfK6dhZc0Tx1KI/gviKL6qvvFs1+bWtaz58uUNnryq6kt5RzOCkPWlVqVX2a/EEBUdU1KrXLf40GoiiFXK///qpoiDXrOgqDR38JB0bw7SoL+ZB9o1RCkQjQ2CBYZKd/+VJxZRRZlqSkKiws0WFxUyCwsKiMy7hUVFhIaCrNQsKkTIsLivwKKigsj8XYlwt/WKi2N4d//uQRCSAAjURNIHpMZBGYiaQPSYyAAABLAAAAAAAACWAAAAApUF/Mg+0aohSIRobBAsMlO//Kk4soosy1JSFRYWaLC4qZBYWFRGZdwqKiwkNBVmoWFSJkWFxX4FFRQWR+LsS4W/rFRb/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////VEFHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU291bmRib3kuZGUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAwNGh0dHA6Ly93d3cuc291bmRib3kuZGUAAAAAAAAAACU=");

function beep() {
    beep_snd.play();
}

  function reloader(image, url, seq) {
    console.log("reloader " + url + seq);
    
    $(image).unbind("load");
    $(image).unbind("error");
    $(image).addClass("reloading");

    var canvas = $(image).siblings('canvas.imagecanvas')[0];
  
    if (!canvas && $(image).hasClass("imagecanvas") && image.naturalWidth && image.naturalHeight) {
      canvas = $("<canvas class='reloader imagecanvas'/>").insertAfter(image)[0];
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      //canvas.getContext("2d").drawImage(image, 0, 0);

      image.style.display = 'none';
        
      register_swipe(canvas);

      $(window).on('DOMContentLoaded load resize scroll', function() {
        if (isElementInViewport(canvas) && !$(image).hasClass("reloading")) {
          reloader(image, url, seq);
        }
      }); 
    }
  
    if (canvas && image.naturalWidth && image.naturalHeight) {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d").drawImage(image, 0, 0);
    }

    if (canvas && !isElementInViewport(canvas)) {
      $(image).removeClass("reloading");
      return;
    }

    var xhr = new XMLHttpRequest();
    xhr.open('GET', url + "?seq=" + seq);
//    xhr.responseType = 'arraybuffer';
    xhr.responseType = 'blob';
    xhr.timeout = 60000;
    xhr.onload = function() {
      if (this.status == 200) {
        seq = Number(this.getResponseHeader('X-seq'));

        //var dataView = new DataView(this.response);
        //var blob = new Blob( [ dataView ], { type: "image/jpeg" } );
        var blob = this.response;
        var imageUrl = URL.createObjectURL( blob );
        
        image.src = imageUrl;
        
        $(image).bind("load", function() {
            URL.revokeObjectURL(this.src);
            reloader(image, url, seq + 1);
        });
        $(image).bind("error", function() {
            URL.revokeObjectURL(this.src);
            reloader(image, url, seq);
        });

      } else {
        console.log("failed " + url);
        setTimeout(function(){
          reloader(image, url, seq);
        }, 4000);
      }
    };
    xhr.ontimeout = function() { 
      console.log("timeout " + url);
      reloader(image, url, seq);
    };
    xhr.onerror = function() { 
      console.log("error " + url);
      setTimeout(function(){
        reloader(image, url, seq);
      }, 4000);
    };
    xhr.send();
  };

  function update_status() {
    $.ajax({type: "GET", url: "status.json", async: false,
      success:function(status) {
        $("#json").text( JSON.stringify(status, null, 2) );

        //alert(JSON.stringify(status));
        $(".ajax_up").each(function() {
          var jsonp = $(this).data('jsonp');
          try {
            var v = jsonp.split(".").reduce( (dict, key) => (key != undefined ? dict[key] : undefined), status);
          } 
          catch(err) {
            return;
          }
          var prefix = $(this).data('prefix');
          if (v == undefined) return;
          if ($(this).is("select")) {
            $(this).children().each(function() {
              var opt = $(this).val();
              var optv = opt;
              if (prefix != undefined) optv = opt.slice(prefix.length);
              if (v == optv) v = opt;
            });
          }
          else if (prefix != undefined) v = prefix + v;
          if ($(this).is(".plot")) {
            plot_data(this, v);
          }
          else if ($(this).is(".ajax_sem")) {
            if (v) $(this).addClass("ajaxrun");
            else $(this).removeClass("ajaxrun");
          }
          else if ($(this).is("span") || $(this).is("option")) {
            $(this).text(v);
          }
          else {
            $(this).removeData('allow_cb')
            if (($(this).val() != v) && !$(this).is(":focus")) {
                $(this).val(v);
            }
            $(this).data('allow_cb', $.now())
          }
          if ($(this).is(".ajax_beep")) {
            if (v >= $(this).data(("beep"))) beep();
          }
        });
      },
      error:function() {
        beep();
      }
    });
  }
 
  function plot_data(placeholder, data) {
    var res = []
    var names = $(placeholder).data('series').split(",");
 //   alert(JSON.stringify(names));
    for (var j = 0; j < names.length; ++j) {
        var cn = names[j];
        if (!(cn in data)) continue;
        if (!$.isArray(data[cn]) ||  data[cn].length == 0) continue;
        var curv = [];
        for (var i = 0; i < data[cn].length; ++i) {
          curv.push([i, data[cn][i]]);
        }
        res.push({ data: curv, label: cn});
    }
 //   alert(JSON.stringify(res));
    $(placeholder).plot(res, {
                        series: {
                                shadowSize: 0   // Drawing is faster without shadows
                        },
                        xaxis: {
                                show: false
                        },
                        legend: {
                                backgroundOpacity: 0
                        }
                        });
  }
 
  function update_transform(transform, zoom, tx, ty, center_x, center_y) {
    var new_zoom = transform.zoom * zoom;
    if (new_zoom < 0.9) new_zoom = 0.9;
    
    var cx = center_x * (1 - new_zoom / transform.zoom);
    var cy = center_y * (1 - new_zoom / transform.zoom);
    transform.zoom = new_zoom;
    transform.x = transform.x + tx + cx;
    transform.y = transform.y + ty + cy;
  }
  function fix_borders(transform, img, box, initial) {
    var bh = box.height();
    var bw = box.width();
    var ih = img.height() * transform.zoom;
    var iw = img.width() * transform.zoom;
  
    if (transform.y > Math.max(0, ih - bh)) transform.y = Math.max(0, ih - bh);
    if (transform.y < Math.min(0, ih - bh)) transform.y = Math.min(0, ih - bh);

    if (transform.x > Math.max(0, bw - iw)) transform.x = Math.max(0, bw - iw);
    if (transform.x < Math.min(0, bw - iw)) transform.x = Math.min(0, bw - iw);

    if (initial) {
      if (transform.x > 0) transform.x = 0;
      if (transform.y < 0) transform.y = 0;
    }

  }
  
  function register_swipe(elem) {

    var transform = { zoom:1.0, x:0, y:0 };
    $(elem).data('transform', transform );
    fix_borders(transform, $(elem), $(elem).parent(".imgbox"), true);
    $(elem).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');


    $(elem).swipe( {
      pinchStatus:function(event, phase, direction, distance , duration , fingerCount, pinchZoom, fingerData) {
        var transform = jQuery.extend({}, $(this).data('transform'));
        var x = fingerData[0].end.x - fingerData[0].start.x;
        var y = fingerData[0].end.y - fingerData[0].start.y;
        var startpos;
        if (phase == 'start') {
          var offset = $(this).offset();
          var height = $(this).height() * transform.zoom;
          startpos = {x: fingerData[0].start.x - offset.left, y: fingerData[0].start.y - offset.top - height};
          $(this).data('startpos', startpos);
        }
        else {
          startpos = $(this).data('startpos');
        }
        update_transform(transform, pinchZoom, x, y, startpos.x, startpos.y)
        fix_borders(transform, $(this), $(this).parent(".imgbox"), false);
        
        if (phase == 'cancel') {
          transform = $(this).data('transform');
        }
        if (phase == 'end') {
            $(this).data('transform', transform);
        }
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
        $(this).data('fingerData', fingerData);
        return true;
      },
      doubleTap:function(event, target) {
        var transform = $(this).data('transform');
        if (transform.zoom != 1.0) {
          transform.zoom = 1.0;
          transform.x = 0;
          transform.y = 0;
        }
        else {
          var img = $(this).parent().children('img')[0];
          var fingerData = $(this).data('fingerData');
          var offset = $(this).offset();
          var height = $(this).height() * transform.zoom;
          
          update_transform(transform, img.naturalWidth / $(this).width(), 0, 0, fingerData[0].start.x - offset.left, fingerData[0].start.y - offset.top - height);
        }
        fix_borders(transform, $(this), $(this).parent(".imgbox"), true);
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
        $('html, body').animate({scrollTop: $(this).parent(".imgbox").offset().top}, 200);
      },
      longTap:function(event, target) {
      	//alert("long tap");
      },
      tap:function(event, target) {
        var transform = $(this).data('transform');
        fix_borders(transform, $(this), $(this).parent(".imgbox"), true);
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
        $('html, body').animate({scrollTop: $(this).parent(".imgbox").offset().top}, 200);
      	//alert("tap");
      },
      pinchThreshold:5,
      threshold:5,
      doubleTapThreshold:500,
      allowPageScroll:"none",
      preventDefaultEvents: true
      
    })
  }


  $(document).ready(function(){
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position){
           var cmd = "gps" + position.coords.latitude + ',' + position.coords.longitude;
           $.ajax({type: "POST", url: "button", data: {cmd: cmd}});
        });
    } 
    
    $("button.ajax").click(function(){
        var btn = $(this);
        var cmd_a = $(this).attr('value').split("!");
        var cmd;
        
        //alert(JSON.stringify(cmd_a));
        if (cmd_a.length == 1) {
            cmd = { cmd: cmd_a[0] };
        }
        else if (cmd_a.length == 2) {
            cmd = { cmd: cmd_a[1], tgt: cmd_a[0] };
        }
        else return;

        btn.addClass("ajaxrun");
        btn.removeClass("ajaxerr");
        
        $.ajax({type: "POST", url: "button", data: cmd,
          success:function() {
            btn.removeClass("ajaxrun");
          },
          error:function() {
            btn.removeClass("ajaxrun");
            btn.addClass("ajaxerr");
          }
        });
    });
    $("select.ajax").change(function(){
        var sel = $(this);
        sel.blur();
        var allow_cb = sel.data('allow_cb')
        if (!allow_cb) return;
        if ($.now() - allow_cb > 30000) return;

        var cmd_a = this.value.split("!");
        //alert(JSON.stringify(cmd_a));
        var cmd;
        if (cmd_a.length == 1) {
            cmd = { cmd: cmd_a[0] }
        }
        else if (cmd_a.length == 2) {
            cmd = { cmd: cmd_a[1], tgt: cmd_a[0] };
        }
        else return;


        sel.addClass("ajaxrun");
        sel.removeClass("ajaxerr");
        $.ajax({type: "POST", url: "button", data: cmd,
          success:function() {
            sel.removeClass("ajaxrun");
          },
          error:function() {
            sel.removeClass("ajaxrun");
            sel.addClass("ajaxerr");
          }
        });
    });

    $("button.reflink").click(function(){
        window.location = this.id;
    });

    $("button.toggle").click(function(){
        var id = '#' + $(this).attr('id') + '_buttons';
        var display = $(id).css('display');
        if (display == 'none')
            display = 'block';
        else
            display = 'none';
        $(id).css('display', display);
    });
    
    $(".viewportbox").swipe( {
       swipeRight:function(event, direction, distance, duration, fingerCount) {
          var link = $("button.prevlink");
          if (link.length > 0) {
            window.location = link[0].id;
          }
        },
       swipeLeft:function(event, direction, distance, duration, fingerCount) {
          var link = $("button.nextlink");
          if (link.length > 0) {
            window.location = link[0].id;
          }
        },
       swipeUp:function(event, direction, distance, duration, fingerCount) {
          $('html, body').animate({scrollTop: $(this).next(".viewportbox").offset().top}, 200);  
        },
       swipeDown:function(event, direction, distance, duration, fingerCount) {
          $('html, body').animate({scrollTop: $(this).prev(".viewportbox").offset().top}, 200);  
        },
      threshold: 100,
      maxTimeThreshold:500,
      allowPageScroll:"none",
      preventDefaultEvents: true
    });


    $(window).on('resize orientationChange', function(event) {
      $("img.reloader").each(function() {
        var elem = this;
        if ($(this).hasClass("imagecanvas")) elem = $(this).siblings('canvas.imagecanvas')[0];
        if (!elem) return;
        var transform = $(elem).data('transform');
        fix_borders(transform, $(elem), $(elem).parent(".imgbox"), true);
        $(elem).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
      });
    });
    
    
    $("img.reloader").bind("load", function() {
        var image = this;
        
        if (! $(image).hasClass("imagecanvas")) {
          register_swipe(image);
        }
        
        var url = image.src;
        reloader(image, url, 1)
    }).bind("error", function() {
        $(this).load();
    }).each(function() {
        if(this.complete) $(this).load();
    });
     
    update_status();
    setInterval(update_status, 5000);
    
    //document.body.addEventListener('touchstart', function(e){ e.preventDefault(); });
    $('html, body').animate({scrollTop: $(".viewportbox").first().offset().top}, 200);
   });
 