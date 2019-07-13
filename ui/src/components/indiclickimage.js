import React from 'react';
import ZoomImage from './zoomimage';
import { INDIContext } from './indi';

export default class INDIClickImage extends React.Component {
  static contextType = INDIContext;
  constructor(props) {
    super(props);
    
    ["xyclick"].forEach(method => this[method] = this[method].bind(this));
  }

  xyclick(x, y) {
    this.context.indi.actionSetProp(this.props.device, this.props.property, {'X': x, 'Y': y});
  }

  render() {
    return ( 
      <ZoomImage src={this.props.src} xyclick={this.xyclick} />
    );
  }
}


