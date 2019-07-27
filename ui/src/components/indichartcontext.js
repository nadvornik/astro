import React from 'react';
import Chart from "react-apexcharts";
import { INDIContext } from './indi';
import pako from "pako";

export default class INDIChartContext extends React.Component {
  static contextType = INDIContext;
  constructor(props, context) {
    super(props, context);

    if (this.props.enable_blob) {
      this.enableBLOB(this.props.path)
    }

  }

  enableBLOB(path) {
    var [device, property, element] = path.split(":");
    this.context.indi.enableBLOB(device, property);
  }
  

  getProp(path) {
    var [device, property, element] = path.split(":");
    if (!this.context.state.entries[device]) return [];
    return [this.context.state.entries[device][property], element]
  }

  render() {
    var [indiprop, element] = this.getProp(this.props.path);
    var history;
    if (this.props.history) {
        history = this.context.state.history[indiprop.device][indiprop.name]
    }
    
    return (
       React.cloneElement(React.Children.only(this.props.children), {indiprop: indiprop, element: element, history: history})
     );
  }
}
