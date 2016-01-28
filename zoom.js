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
  
    if (!canvas && image.naturalWidth && image.naturalHeight) {
      canvas = $("<canvas class='imagecanvas'/>").insertAfter(image)[0];
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d").drawImage(image, 0, 0);

      image.style.display = 'none';
        
      register_swipe(canvas);

      $(window).on('DOMContentLoaded load resize scroll', function() {
        if (isElementInViewport(canvas) && !$(image).hasClass("reloading")) {
          reloader(image, url, seq);
        }
      }); 
    }
  
    if (canvas && image.naturalWidth && image.naturalHeight) canvas.getContext("2d").drawImage(image, 0, 0);

    if (canvas && !isElementInViewport(canvas)) {
      $(image).removeClass("reloading");
      return;
    }

    var xhr = new XMLHttpRequest();
    xhr.open('GET', url + "?seq=" + seq);
//    xhr.responseType = 'arraybuffer';
    xhr.responseType = 'blob';
    xhr.timeout = 4000;
    xhr.onload = function() {
      if (this.status == 200) {
        seq = Number(this.getResponseHeader('X-seq'));

        //var dataView = new DataView(this.response);
        //var blob = new Blob( [ dataView ], { type: "image/jpeg" } );
        var blob = this.response;
        var imageUrl = URL.createObjectURL( blob );
        
        image.src = imageUrl;
        
        $(image).bind("load", function() {
            reloader(image, url, seq + 1);
        });
        $(image).bind("error", function() {
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
        //alert(JSON.stringify(status));
        $(".ajax_up").each(function() {
          var jsonp = $(this).data('jsonp');
          var v = jsonp.split(".").reduce( (dict, key) => (key != undefined ? dict[key] : undefined), status);
          var prefix = $(this).data('prefix');
          if (v == undefined) return;
          if (prefix != undefined) v = prefix + v;
          $(this).val(v);
        });
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
  
    $("#txt2").text( JSON.stringify(iw + ' ' + bw) );
    if (transform.y > Math.max(0, ih - bh)) transform.y = Math.max(0, ih - bh);
    if (transform.y < Math.min(0, ih - bh)) transform.y = Math.min(0, ih - bh);

    if (transform.x > Math.max(0, bw - iw)) transform.x = Math.max(0, bw - iw);
    if (transform.x < Math.min(0, bw - iw)) transform.x = Math.min(0, bw - iw);

    if (initial) {
      if (transform.x > 0) transform.x = 0;
      if (transform.y < 0) transform.y = 0;
    }

  }
  
  function register_swipe(canvas) {

    var transform = { zoom:1.0, x:0, y:0 };
    $(canvas).data('transform', transform );
    fix_borders(transform, $(canvas), $(canvas).parent(".imgbox"), true);
    $(canvas).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');


    $(canvas).swipe( {
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
  
//            var err = new Error();
//  	    $("#txt2").text( err.stack);
            
        }
        if (phase == 'end') {
            $(this).data('transform', transform);
        }
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
        $(this).data('fingerData', fingerData);
//        $("#txt2").text( JSON.stringify(startpos) );
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
          var img = $(this).siblings('img.imagecanvas')[0];
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
    $("button.ajax").click(function(){
        var btn = $(this);
        btn.addClass("ajaxrun");
        btn.removeClass("ajaxerr");
        $.ajax({type: "POST", url: "button", data: {key: $(this).attr('id')},
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
        var value = this.value;
        if ( sel.prop("selectedIndex") == 0 ) return;
        sel.addClass("ajaxrun");
        sel.removeClass("ajaxerr");
        $.ajax({type: "POST", url: "button", data: {key: value},
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
      $("canvas.imagecanvas").each(function() {
        var transform = $(this).data('transform');
        fix_borders(transform, $(this), $(this).parent(".imgbox"), true);
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
      });
    });
    
    
    $("img.imagecanvas").bind("load", function() {
        var image = this;
        
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
 