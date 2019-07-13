import React from 'react';
import Chart from "react-apexcharts";
import update from 'immutability-helper';
import INDIChart from './indichart';

export default class FocusChart extends INDIChart {

  constructor(props, context) {
    super(props, context);

    this.enableBLOB(this.props.path);
  }
  
  
  render() {
    var focus_data = this.getJSONBLOB(this.props.path);
    if (!focus_data)  return <div/>;
    
    var series = ["v_curve", "v_curve_s"].filter(name => {
      return focus_data[name];
    }).map(name => ({
      name: name,
      data: focus_data[name].map((v, i) => ([i, v]))
    }));

    series = series.concat( ["v_curve2", "v_curve2_s"].filter(name => {
      return focus_data[name];
    }).map(name => ({
      name: name,
      data: focus_data[name].map((v, i) => ([i + (focus_data.hyst || 0), v]))
    })));

    var xannotation = [];
    
    if (focus_data.xmin) {
      series.push({
        name: 'V',
        data: [
          [0, focus_data.c1], 
          [focus_data.xmin, focus_data.xmin * focus_data.m1 + focus_data.c1],
          [focus_data.xmin * 2, focus_data.xmin * 2 * focus_data.m2 + focus_data.c2]
        ]
      });
      xannotation = [{x:focus_data.xmin}];

    }

    var options = update(this.state, {
      annotations: {xaxis: {$set: xannotation  }},
    })


    return (
      <Chart
        options={options}
        series={series}
        type="line"
        height="300"
      />
    );
  }
}
