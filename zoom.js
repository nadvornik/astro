function isElementInViewport(el) {
    var rect = el.getBoundingClientRect();

    return rect.bottom > 0 &&
        rect.right > 0 &&
        rect.left < (window.innerWidth || document. documentElement.clientWidth) /*or $(window).width() */ &&
        rect.top < (window.innerHeight || document. documentElement.clientHeight) /*or $(window).height() */;
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
    $.ajax({type: "GET", url: "status.json",
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
          
          if ($(this).is("#plot_hfr")) {
            plot_hfr_data(this, v);
          }
          else if ($(this).is(".ajax_sem")) {
            if (v) $(this).addClass("ajaxrun");
            else $(this).removeClass("ajaxrun");
          }
          else if ($(this).is("span")) {
            $(this).text(v);
          }
          else {
            $(this).data('no_cb', 1)
            $(this).val(v);
            $(this).removeData('no_cb')
          }
        });
      }
    });
  }
 
  var plot_hfr;
  function plot_hfr_data(canvas, data) {
    var res = []
    var names = ['v_curve', 'v_curve_s', 'v_curve2', 'v_curve2_s'];
    for (var j = 0; j < names.length; ++j) {
        var cn = names[j];
        if (!(cn in data)) continue;
        if (!$.isArray(data[cn]) ||  data[cn].length == 0) continue;
        var curv = [];
        for (var i = 0; i < data[cn].length; ++i) {
          curv.push([i, data[cn][i]]);
        }
        res.push(curv);
    }
    plot_hfr.setData(res);
    plot_hfr.setupGrid();
    plot_hfr.draw();

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
           cmd = "gps" + position.coords.latitude + ',' + position.coords.longitude;
           $.ajax({type: "POST", url: "button", data: {cmd: cmd}});
        });
    } 
    
    $("button.ajax").click(function(){
        var btn = $(this);
        if (btn.data('no_cb')) return;
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
        if (sel.data('no_cb')) return;

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
    
    plot_hfr = $.plot($("#plot_hfr"), [ ], {
                        series: {
                                shadowSize: 0   // Drawing is faster without shadows
                        },
                        xaxis: {
                                show: false
                        }});
  });
 