import React from 'react';
import Chart from "react-apexcharts";
import pako from "pako";

export default class INDIChart extends React.PureComponent {
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

  historyToSeries() {
  
    var series = Object.values(this.props.indiprop.elements).map(e => ({
      name: e.name, 
      data: this.props.history.map(he => (he[e.i]))
    }));
    return series;
  }

  getJSONBLOB() {
    const indiprop = this.props.indiprop;
    const element = this.props.element;
    var blob_data
    
    try {
      if (indiprop.elements[element].format === '.json') {
      
          blob_data = JSON.parse(indiprop.elements[element].value);
      }
      else if (indiprop.elements[element].format === '.json.z') {
          blob_data = JSON.parse(pako.inflate(indiprop.elements[element].value, { to: 'string' }));
      }
    } catch (error) {
      console.log(indiprop);
      console.log(error);
      return;
    }
    console.log(blob_data);
    return blob_data;
  }

  blobToSeries() {
    var data = this.getJSONBLOB();
    if (!data) return;
    
    var series = Object.keys(data).map(name => ({
      name: name, 
      data: data[name]
    }));
    console.log(series);
    return series;
  }

  render() {
    var series;
    if (this.props.history) 
        series = this.historyToSeries();
    else
        series = this.blobToSeries();

    if (! series) return <div/>;

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
