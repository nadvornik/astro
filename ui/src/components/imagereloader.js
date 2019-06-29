

import React from 'react';
import ZoomImage from './zoomimage';
import VisibilitySensor from 'react-visibility-sensor';

export default class ImageReloader extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      src: props.src
    };
    this.seq = 0;
    this.reloading = false;
    this.visible = true;

    this.onVisibilityChange = this.onVisibilityChange.bind(this);
  }

  reload() {
    if (this.reloading) return;

    this.reloading = true;
    fetch(this.props.src + "?seq=" + (this.seq + 1))
      .then(res => {
        if (res.status === 200) {
          //console.log(res);
          this.seq = Number(res.headers.get('X-seq'));
          //console.log('seq', this.seq);
          return res.blob();
        }
        throw new Error("Response not OK");
      })
      .then( blob => {
        var imageUrl = URL.createObjectURL( blob );
        this.setState({src: imageUrl});
        this.reloading = false;
        if (this.visible) {
            this.reload();
        }
      })
      .catch( e => {
        this.reloading = false;
        console.log("error ", e);
        setTimeout(() => {
          this.reload();
        }, 4000);
      });
  }

  componentDidMount() {
    this.reload();
  }

  onVisibilityChange(isVisible) {
    this.visible = isVisible;
    if (isVisible && !this.reloading) {
      this.reload();
    }
  }

  render() {
    return ( 
      <VisibilitySensor onChange={this.onVisibilityChange} partialVisibility={true}>
        <ZoomImage src={this.state.src} />
      </VisibilitySensor>
    );
  }
}


