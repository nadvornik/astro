import React from 'react';
import { INDIContext } from './indi';

export default class INDIChartContext extends React.Component {
  static contextType = INDIContext;

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
