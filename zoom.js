 
  function update_transform(transform, zoom, tx, ty, center_x, center_y) {
    var new_zoom = transform.zoom * zoom;
    if (new_zoom < 0.9) new_zoom = 0.9;
    
    var cx = center_x * (1 - new_zoom / transform.zoom);
    var cy = center_y * (1 - new_zoom / transform.zoom);
        $("#txt").text( center_x );
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
    
    $(".image").swipe( {
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
          var img = $(this)[0];
          var fingerData = $(this).data('fingerData');
          var offset = $(this).offset();
          var height = $(this).height() * transform.zoom;
          
          update_transform(transform, img.naturalWidth / img.clientWidth, 0, 0, fingerData[0].start.x - offset.left, fingerData[0].start.y - offset.top - height);
          

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

    
    $(".image").each(function() {
      var transform = { zoom:1.0, x:0, y:0 };
      $(this).data('transform', transform );
      fix_borders(transform, $(this), $(this).parent(".imgbox"), true);
      $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
    });
    
    $(window).on('resize orientationChange', function(event) {
      $(".image").each(function() {
        var transform = $(this).data('transform');
        fix_borders(transform, $(this), $(this).parent(".imgbox"), true);
        $(this).css('transform', ' translate(' + transform.x + 'px, ' + transform.y + 'px)  scale(' + transform.zoom + ')');
      });
    });
    
    //document.body.addEventListener('touchstart', function(e){ e.preventDefault(); });
    $('html, body').animate({scrollTop: $(".viewportbox").first().offset().top}, 200);  
  });
 