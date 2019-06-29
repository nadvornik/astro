

import React from 'react';
import './zoomimage.css';


export default class ZoomImage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      img_w: 1,
      img_h: 1,
      canvas_w: 1,
      canvas_h: 1,
      x: 0,
      y: 0,
      scale: 1,
      fit: true
    };

    this.dragging = false;
    this.doubleclick_timer = false;


    this.onImgLoad = this.onImgLoad.bind(this);
    this.updateCanvas = this.updateCanvas.bind(this);
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onWheel = this.onWheel.bind(this);
  }
  
  fitWindow(center_x, center_y) {
    var new_scale = Math.min(this.state.canvas_w / this.state.img_w, this.state.canvas_h / this.state.img_h);
    this.updateTransform(new_scale, 0, 0, center_x, center_y, {fit: true});
  }

  updateTransform(new_scale, tx, ty, center_x, center_y, update) {
    var new_s = {...this.state, ...update}
//    if (new_scale < 0.9) new_scale = 0.9;
    
    var cx = center_x * (1 - new_scale / new_s.scale);
    var cy = center_y * (1 - new_scale / new_s.scale);
    new_s.scale = new_scale;
    new_s.x = new_s.x + tx + cx;
    new_s.y = new_s.y + ty + cy;
    this.fixBorders(new_s);
  }

  fixBorders(update) {
    var new_s = {...this.state, ...update}
    
    if (new_s.fit) {
      new_s.scale = Math.min(new_s.canvas_w / new_s.img_w, new_s.canvas_h / new_s.img_h);
    }
    
    var ch = new_s.canvas_h;
    var cw = new_s.canvas_w;
    var ih = new_s.img_h * new_s.scale;
    var iw = new_s.img_w * new_s.scale;
  
    if (new_s.y > Math.max(0, ch - ih)) new_s.y = Math.max(0, ch - ih);
    if (new_s.y < Math.min(0, ch - ih)) new_s.y = Math.min(0, ch - ih);

    if (new_s.x > Math.max(0, cw - iw)) new_s.x = Math.max(0, cw - iw);
    if (new_s.x < Math.min(0, cw - iw)) new_s.x = Math.min(0, cw - iw);

    if (this.state.img_w === 1 || this.state.canvas_w === 1 ) {
      if (new_s.x > 0) new_s.x = 0;
      if (new_s.y > 0) new_s.y = 0;
    }
    this.setState(new_s);
  }


  onImgLoad({target:img}) {
    this.fixBorders({ img_h: img.naturalHeight, img_w: img.naturalWidth });
  }

  updateCanvas() {
    if (this.state.canvas_h !== this.imgbox.clientHeight ||
        this.state.canvas_w !== this.imgbox.clientWidth) {
      this.fixBorders({ canvas_h: this.imgbox.clientHeight, canvas_w: this.imgbox.clientWidth });
    }
  }

  componentDidMount() {
    this.updateCanvas();
    window.addEventListener("resize", this.updateCanvas);
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.updateCanvas);
    window.removeEventListener('mousemove', this.onMouseMove)
    window.removeEventListener('mouseup', this.onMouseUp)
  }

  componentDidUpdate() {
    this.updateCanvas();
  }


  onMouseDown(e) {
    if (e.button !== 0) return;
    
    var bbox = this.img.getBoundingClientRect();
    this.startpos = {
      x: e.pageX - bbox.left,
      y: e.pageY - bbox.top
    };

    if (this.doubleclick_timer) {
      if (this.state.scale === 1) {
        this.fitWindow(this.startpos.x, this.startpos.y);
      }
      else {
        this.updateTransform(1, 0, 0, this.startpos.x, this.startpos.y, {fit: false});
      }
    
      this.doubleclick_timer = false;
      e.stopPropagation();
      e.preventDefault();
      console.log("doubleclick");
      return;
    }
    
    
    window.addEventListener('mousemove', this.onMouseMove)
    window.addEventListener('mouseup', this.onMouseUp)
    
    e.stopPropagation();
    e.preventDefault();
    console.log("onMouseDown", this.startpos);
    this.dragging = true;
  }

  onMouseUp(e) {
    this.dragging = false;
    window.removeEventListener('mousemove', this.onMouseMove)
    window.removeEventListener('mouseup', this.onMouseUp)

    var bbox = this.img.getBoundingClientRect();

    var x = e.pageX - bbox.left;
    var y = e.pageY - bbox.top;

    if (x === this.startpos.x && y === this.startpos.y) {
      this.doubleclick_timer = true;
      setTimeout(() => {
        this.doubleclick_timer = false;
      }, 400);
    }

    e.stopPropagation();
    e.preventDefault();
  }

  onMouseMove(e) {
    if (!this.dragging) return;
    var bbox = this.img.getBoundingClientRect();

    var x = e.pageX - bbox.left;
    var y = e.pageY - bbox.top;
    console.log("onMouseMove", x - this.startpos.x, y - this.startpos.y);
    this.updateTransform(this.state.scale, x - this.startpos.x, y - this.startpos.y, this.startpos.x, this.startpos.y);
    e.stopPropagation();
    e.preventDefault();
  }

  onWheel(e) {
    var bbox = this.img.getBoundingClientRect();

    var x = e.pageX - bbox.left;
    var y = e.pageY - bbox.top;

    if (e.deltaY > 0)
      this.updateTransform(this.state.scale * 1.1, 0, 0, x, y, {fit: false});
    else
      this.updateTransform(this.state.scale * 0.9, 0, 0, x, y, {fit: false});

    e.stopPropagation();
    e.preventDefault();
  }


  render() {
    const style = { 
        transform: `translate(${this.state.x}px, ${this.state.y}px) scale(${this.state.scale}) `,
        transformOrigin: '0% 0%'
//        maxWidth: '100%',
//        maxHeight: '100%'
    };

    return (
      <div className='imgbox' ref={ (div) => this.imgbox = div}>
        <img 
          src={ this.props.src }
          style={ style }
          onLoad={ this.onImgLoad }
          onMouseDown={ this.onMouseDown }
          onWheel={ this.onWheel }
          ref={ (img) => this.img = img}
        />
      </div>
    );
  }
}

