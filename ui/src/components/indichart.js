import React from 'react';
import Chart from "react-apexcharts";
import update from 'immutability-helper';

export default class INDIChart extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      options: {
        chart: {
          type: "line",
          stacked: false,
          animations: {
            enabled: false
          }
        },
        xaxis: {
          type: "numeric",
          axisTicks: {
            show: false
          }
        },
        yaxis: {
          forceNiceScale: true,
          decimalsInFloat: 0,
          tickAmount: 7
        },
        stroke: {
          width: 1,
          curve: 'straight'
        },
        dataLabels: {
          enabled: false
        }
          
      },
      series: [
      ]
    };
    
    ['addValue'].forEach(method => this[method] = this[method].bind(this));
    
    console.log(this.props);
    if (this.props.registerIndiCb) this.props.registerIndiCb('Guider', 'offset', this.addValue);
  }

  addValue(indiprop) {
    console.log("add value", indiprop);
    
    if (this.state.series.length === 0) {
      this.setState({series: Object.values(indiprop.elements).map(e => ({name: e.name, data: []}))});
    }
    
    Object.values(indiprop.elements).forEach((e, i) => {
      this.setState(prevState => (
        update(prevState, {series: {[i]: {data: {$push: [e.value]}}}})
      ))
    });
  }

  render() {
    return (
      <Chart
        options={this.state.options}
        series={this.state.series}
        type="line"
        height="300"
      />
    );
  }
}
