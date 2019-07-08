import React from 'react';
import logo from './logo.svg';
import './App.css';
import INDI from './components/indi';
import ImageReloader from './components/imagereloader';
import ZoomImage from './components/zoomimage';


class App extends React.Component {
  constructor(props) {
    super(props);

    this.xyclick = this.xyclick.bind(this);
    this.indi = React.createRef();
  }

  xyclick(x, y) {
    this.indi.current.actionSetProp('Navigator', 'zoom_pos', {'X': x, 'Y': y});
  }

  render() {
    return (
      <div className="App">
        <div className="Images">
          <div className='viewport' >
            <ImageReloader src='navigator.jpg'>
              <ZoomImage xyclick={this.xyclick}/>
            </ImageReloader>
          </div>
          <div className='viewport' >
            <ImageReloader src='guider.jpg'>
              <ZoomImage/>
            </ImageReloader>
          </div>
          <div className='viewport' >
            <ImageReloader src='full_res.jpg'>
              <ZoomImage/>
            </ImageReloader>
          </div>
          <div className='viewport' >
            <ImageReloader src='polar.jpg'>
              <ZoomImage/>
            </ImageReloader>
          </div>
        </div>
        <div className="INDIwrapper">
          <INDI wsurl='websocket' ref={this.indi}/>
        </div>
      </div>
    );
  }
}

export default App;
