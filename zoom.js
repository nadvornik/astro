 

  function reloader(canvas, url, seq) {
    //alert("reloader " + url + " " + seq);
    var image = $(canvas).siblings('img.imagecanvas')[0];

    var xhr = new XMLHttpRequest();
    xhr.open('GET', url + "?seq=" + seq);
//    xhr.responseType = 'arraybuffer';
    xhr.responseType = 'blob';
    xhr.timeout = 5000;
    xhr.onload = function() {
      if (this.status == 200) {
        var newseq = this.getResponseHeader('X-seq');
        seq = Math.max(seq, Number(newseq))

        //var dataView = new DataView(this.response);
        //var blob = new Blob( [ dataView ], { type: "image/jpeg" } );
        var blob = this.response;
        var imageUrl = URL.createObjectURL( blob );
        
        image.src = imageUrl;
        
        image.onload = function() {
            canvas.getContext("2d").drawImage(image, 0, 0);
            reloader(canvas, url, seq + 1)
        };


      } else {
//          reloader(canvas, url, seq)
      }
    };
    xhr.ontimeout = function() { 
      console.log("reload");
      reloader(canvas, url, seq);
    };
    xhr.send();
  };
  
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
            sel.prop("selectedIndex",0);
            sel.children().first().text(value);
          },
          error:function() {
            sel.removeClass("ajaxrun");
            sel.addClass("ajaxerr");
            sel.prop("selectedIndex",0);
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
    
    
    $("img.imagecanvas").one("load", function() {
        var image = this;
        
        var canvas = document.createElement("canvas");
        
        var canvas = $("<canvas class='imagecanvas'/>").insertAfter(image)[0];
	canvas.width = image.naturalWidth;
	canvas.height = image.naturalHeight;
	canvas.getContext("2d").drawImage(image, 0, 0);
        
        image.style.display = 'none';
        
        register_swipe(canvas);
        
        var url = image.src;
        reloader(canvas, url, 1)
    }).each(function() {
      if(this.complete) $(this).load();
    });
     

    //document.body.addEventListener('touchstart', function(e){ e.preventDefault(); });
    $('html, body').animate({scrollTop: $(".viewportbox").first().offset().top}, 200);  
  });
 