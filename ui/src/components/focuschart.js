import React from 'react';
import Chart from "react-apexcharts";
import update from 'immutability-helper';

export default class FocusChart extends React.Component {
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
          width: 2,
          curve: 'straight'
        },
        dataLabels: {
          enabled: false
        },
        grid: {
          show: false
        },
        annotations: {
          xaxis: [
          ]
        }
        
          
      },
      series: [
      ]
    };
    
    ['addValue'].forEach(method => this[method] = this[method].bind(this));
    
    console.log(this.props);
    if (this.props.registerIndiCb) this.props.registerIndiCb('Navigator', 'focus_data', this.addValue);
  }

  addValue(indiprop) {
    //console.log("add value", indiprop);
    var focus_data
    
    try {
      focus_data = JSON.parse(indiprop.elements.focus_data.value);
    } catch (error) {
      console.log(error);
      return;
    }
    
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
//    console.log(series);

    this.setState(prevState => (
      update(prevState, {
        options: {annotations: {xaxis: {$set: xannotation  }}},
        series: {$set: series}
      })
    ));
    
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
