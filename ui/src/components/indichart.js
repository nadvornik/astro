import React from 'react';
import Chart from "react-apexcharts";
import { INDIContext } from './indi';

export default class INDIChart extends React.Component {
  static contextType = INDIContext;
  constructor(props, context) {
    super(props, context);

    this.state = {
        chart: {
          type: "line",
          stacked: false,
          animations: {
            enabled: false
          },
          toolbar: {
            show: false
          }
        },
        tooltip: {
          enabled: false
        },
        xaxis: {
          type: "numeric",
          axisTicks: {
            show: false
          }
        },
        yaxis: {
          forceNiceScale: true,
          decimalsInFloat: 2
          //tickAmount: 7
        },
        stroke: {
          width: 1,
          curve: 'straight'
        },
        grid: {
          show: true,
          strokeDashArray: 5
        },
        dataLabels: {
          enabled: false
        },
        annotations: {
        },
        ...props.options
    };
    
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

  historyToSeries(indiprop) {
    var series = Object.values(indiprop.elements).map(e => ({
      name: e.name, 
      data: this.context.state.history[indiprop.device][indiprop.name].map(he => (he[e.i]))
    }));
    return series;
  }

  getJSONBLOB(path) {
    const [indiprop, element] = this.getProp(path);
    var blob_data
    
    try {
      blob_data = JSON.parse(indiprop.elements[element].value);
    } catch (error) {
      console.log(indiprop);
      console.log(error);
      return;
    }
    return blob_data;
  }

  render() {
    var [indiprop, element] = this.getProp(this.props.path);
    var series = this.historyToSeries(indiprop);

    return (
      <Chart
        options={this.state}
        series={series}
        type="line"
        height="300"
      />
    );
  }
}
