import React from 'react';
import Chart from "react-apexcharts";
import update from 'immutability-helper';
import INDIChart from './indichart';

export default class HistogramChart extends INDIChart {
  constructor(props) {
    super(props);
    this.state = update(this.state, {
      yaxis: {$merge: {logarithmic: true, yMin: 1}  }
    });
  }

  render() {
    var data = this.getJSONBLOB();
    if (!data)  return <div/>;
    
    var series = ["histogram"].filter(name => {
      return data[name];
    }).map(name => ({
      name: name,
      data: data[name].map(v => Math.max(v, 1))
    }));
    return (
      <Chart
        options={this.state}
        series={series}
        type="bar"
        height="300"
      />
    );
  }
}
