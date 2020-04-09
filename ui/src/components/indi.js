

import React from 'react';
import update from 'immutability-helper';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import "react-tabs/style/react-tabs.css";

import './indi.css';

const XmlReader = require('xml-reader');

update.extend('$auto', function(value, object) {
  return object ?
    update(object, value):
    update({}, value);
});

function format_num(v, format) {
  function round_pad(num, prefix, len) {
    num = String(num);
    var plen = num.indexOf('.');
    if (plen < 0) plen = num.length;
    if (plen < len) {
      return prefix.repeat(len - plen) + num;
    }
    else {
      return num;
    }
  }

  function to_dms(v) {
    var deg = Math.floor(v / 60 / 60 / 100);
    var min = (v - deg * 60 * 60 * 100) / 60 / 100;
    var fmin = Math.floor(min);
    var sec = (min - fmin) * 60;
    return [deg, min, fmin, sec];
  }

  var re_res = /%(0?)([0-9]*)\.?([0-9]*)([a-zA-Z])/g.exec(format);
  if (re_res === null) return v;
  var [match, pad, len, prec, f] = re_res

//  if (pad === '') pad = ' ';

  v = parseFloat(v);
  var sign = '';
  if (v < 0) {
    sign = '-';
    v = -v;
    len = len - 1;
  }

  var out;
  var deg, min, fmin, sec;

  switch (f) {
    case 'd':
      out = parseInt(v, 10);
      break;
    case 'f':
    case 'g':
      out = parseFloat(v);
      if (prec !== '') out = out.toFixed(prec)
      break;
    case 'm':

      switch (prec) {
        case '9':
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60 * 60 * 100))
          out = deg + ':' + round_pad(fmin, '0', 2) + ':' + round_pad(sec.toFixed(2), '0', 2);
          break;
        case '8':
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60 * 60 * 10) * 10)
          out = deg + ':' + round_pad(fmin, '0', 2) + ':' + round_pad(sec.toFixed(1), '0', 2);
          break;
        case '6':
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60 * 60) * 100)
          out = deg + ':' + round_pad(fmin, '0', 2) + ':' + round_pad(sec.toFixed(0), '0', 2);
          break;
        case '5':
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60 * 10) * 60 * 10)
          out = deg + ':' + round_pad(min.toFixed(1), '0', 2);
          break;
        case '3':
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60) * 60 * 100)
          out = deg + ':' + round_pad(min.toFixed(0), '0', 2);
          break;
        default:
          [deg, min, fmin, sec] = to_dms(Math.round(v * 60) * 60 * 100)
          out = deg + ':' + round_pad(min.toFixed(0), '0', 2);
      }
      break;
    default:
      out = v;
  }
  out = sign + round_pad(out, pad, len);
  return out
}

function parse_num(s) {
  var comp = s.match(/^ *(-?\d+(\.\d+)?) *((hr)|h|(deg)|º|:)? *(\d+(\.\d+)?)? *((min)|m|'|:)? *(\d+(\.\d+)?)? *((sec)|s|"|:)? *$/); //'
  if (comp == null) return '';
  
  var mul = 1.0;
  var res = 0;

  var num = parseFloat(comp[1]);
  res += mul * num;
  if (comp[1].startsWith('-'))
    mul = -mul;
  mul /= 60.0;

  if (comp[6] !== undefined) {
    num = parseFloat(comp[6]);
    if (num >= 60.0) return '';
    res += mul * num;
    mul /= 60.0;
  }
  
  if (comp[10]) {
    num = parseFloat(comp[10]);
    if (num >= 60.0) return '';
    res += mul * num;
  }

  return res;
}

function INDIvectorName(props) {
  return (
    <span className='INDIvector_name'>
       <span className={`INDIlight INDIlight_${ props.state }`}/>
       {props.label + ' '}
    </span>
  );
}

function INDIButton(props) {
  var id = `id_${props.name}_${props.vector_name}_${props.device}`;
  return (
    <div className={`INDIvector_item INDIvector_item_button ${props.name} ${props.vector_name}`}>
      <input type="checkbox" id={id} checked={props.value === 'On'} name={id} onChange={props.onChange}/>
      <label htmlFor={id}>
        {props.label || props.name}
      </label>
    </div>
  );
}

function INDIValue(props) {
  return (
    <span className={`INDIvector_item ${props.perm!=='ro' ? 'INDIvector_editable' : ''} ${props.name} ${props.vector_name}`}>
      <span className={`INDIvector_desc`}>{(props.label || props.name) + ' :'}</span>
      <input type='text' value={props.value} readOnly />
      {props.perm!=='ro' && <input
        type='text'
        className={`INDIvector_item_new_value ${props.error || false ? 'INDIvector_item_new_value_err' : ''}`}
        value={props.new_value || ''}
        onChange={props.onChange}
        onPaste={(e) => {e.target.value = ''; return true;}}
      /> }
      {props.submit_show && <button
        className='INDIvector_item_set'
        onClick={props.onSubmit}
        disabled={props.submit_disabled}
      >set</button> }
    </span>
  );
}

class INDIproperty extends React.PureComponent {

  constructor(props) {
    super(props);
    this.state = {
      raw: {}, 
      parsed: {}
    };
    
    Object.values(this.props.elements).forEach(e => {
      this.state.raw[e.name] = (this.props.type === "NumberVector") ? format_num(e.value, e.format) : e.value;
      this.state.parsed[e.name] = e.value;
    });
    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(e, value) {
    var parsed;
    if (this.props.type === "NumberVector") {
      parsed = parse_num(value);
    }
    else {
      parsed = value;
    }
    this.setState(prevState => ({raw: {...prevState.raw, [e.name]: value}, parsed: {...prevState.parsed, [e.name]: parsed}}));
  }

  isValid() {
    var name;
    for (name in this.state.raw) {
      if (this.state.raw[name] !== '' && this.state.parsed[name] === '') {
        return false;
      }
    }
    return true;
  }

  render() {
    if (this.props.type === 'SwitchVector') return (
      <div className={`INDIvector ${this.props.name}`}>
        <INDIvectorName state={ this.props.state } label={this.props.label || this.props.name}/>
        {Object.values(this.props.elements).map(e => (
          <INDIButton vector_name={this.props.name} name={e.name} label={e.label} value={e.value} key={e.name} device={this.props.device}
             onChange={() => this.props.indi.actionSetProp(this.props.device, this.props.name, {[e.name]: (e.value === 'On'? "Off" : "On")})}
          />
        ))}
      </div>
    );
    else if (this.props.type === 'LightVector') return (
      <div className={`INDIvector ${this.props.name}`}>
        <INDIvectorName state={ this.props.state } label={this.props.label || this.props.name}/>
        {Object.values(this.props.elements).map(e => (
          <span className={`INDIvector_item`} key={e.name}>
            <span className={`INDIlight INDIlight_${ e.value }`}/>
            {e.label || e.name}
          </span>
        ))}
      </div>
    );
    else if (this.props.type === 'TextVector') return (
      <div className={`INDIvector ${this.props.name}`}>
        <INDIvectorName state={ this.props.state } label={this.props.label || this.props.name}/>
        {Object.values(this.props.elements).map((e, index) => (
          <INDIValue vector_name={this.props.name} name={e.name} label={e.label} key={e.name} perm={this.props.perm} 
            value={e.value}
            new_value={this.state.raw[e.name] || ''}
            error={this.state.raw[e.name] !== '' && this.state.parsed[e.name] === ''}
            onChange={(event) => this.handleChange(e, event.target.value)}
            submit_show={this.props.perm!=='ro' && index === 0}
            submit_disabled={!this.isValid()}
            onSubmit={() => this.props.indi.actionSetProp(this.props.device, this.props.name, this.state.parsed)}
          />
          ))}
      </div>
    );
    else if (this.props.type === 'NumberVector') {
      return (<div className={`INDIvector ${this.props.name}`}>
        <INDIvectorName state={ this.props.state } label={this.props.label || this.props.name}/>
        {Object.values(this.props.elements).map((e, index) => (
          <INDIValue vector_name={this.props.name} name={e.name} label={e.label} key={e.name} perm={this.props.perm} 
            value={format_num(e.value, e.format)}
            new_value={this.state.raw[e.name] || ''}
            error={this.state.raw[e.name] !== '' && this.state.parsed[e.name] === ''}
            onChange={(event) => this.handleChange(e, event.target.value)}
            submit_show={this.props.perm!=='ro' && index === 0}
            submit_disabled={!this.isValid()}
            onSubmit={() => this.props.indi.actionSetProp(this.props.device, this.props.name, this.state.parsed)}
          />
          ))}
      </div>
      );
    }
    else {
      return <div className={`INDIvector ${this.props.name}`}/>
    }
  }
}


function INDIgroup(props) {
  return (
    <div className={`INDIgroup ${props.name.replace(/\s+/g, '')}`}>
        <h3 className='INDIgroup_name'>{props.name}</h3>
        {(props.extensions || []).filter(ext => (ext.props.tab === `${props.device}:${props.name}`)).map((ext, i) => <div className='INDIextension' key={`_ext${i}`}>{ext}</div>)}
        {Object.keys(props.vec).map(vec => React.createElement(INDIproperty, {...props.vec[vec], indi: props.indi, key: vec}, null)) }
    </div>
  );
}

class INDIdevice extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      tabIndex: 0 
    };
  }
  render() {
    return (
      <div className='INDIdevice'>
        <Tabs selectedIndex={this.state.tabIndex} onSelect={tabIndex => this.setState({ tabIndex })}>
          <TabList>
            {Object.keys(this.props.groups).map(group =>
              <Tab key={group}>{group}</Tab>
            )}
          </TabList>
          
          <h2>{this.props.name}</h2>
          {Object.keys(this.props.groups).map(group => 
            <TabPanel key={group}>
              <INDIgroup
                vec={this.props.groups[group]}
                indi={this.props.indi}
                key={group}
                name={group}
                device={this.props.name}
                extensions={this.props.extensions}
              />
            </TabPanel>
          )}
        </Tabs>
      </div>
    );
  }
}

function INDIMessages(props) {
  return (
    <div className='INDIMessages'>
      <h2>Messages</h2>
      <div>
        {Object.values(props.messages).map((e, index) => (
          <div className='INDIMessage' key={index}>{e}</div>
        ))}
      </div>
    </div>
  );
}

export const INDIContext = React.createContext({});

var beep_snd = new Audio("data:audio/mp3;base64,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qJrx4KOQjahZxWbcZgztj2c49nKmkId44S71j0c8eV9yDK6uPRzx5X18eDvjvQ6yKo9ZSS6l//8elePK/Lf//IInrOF/FvDoADYAGBMGb7FtErm5MXMlmPAJQVgWta7Zx2go+8xJ0UiCb8LHHdftWyLJE0QIAIsI+UbXu67dZMjmgDGCGl1H+vpF4NSDckSIkk7Vd+sxEhBQMRU8j/12UIRhzSaUdQ+rQU5kGeFxm+hb1oh6pWWmv3uvmReDl0UnvtapVaIzo1jZbf/pD6ElLqSX+rUmOQNpJFa/r+sa4e/pBlAABoAAAAA3CUgShLdGIxsY7AUABPRrgCABdDuQ5GC7DqPQCgbbJUAoRSUj+NIEig0YfyWUho1VBBBA//uQZB4ABZx5zfMakeAAAAmwAAAAF5F3P0w9GtAAACfAAAAAwLhMDmAYWMgVEG1U0FIGCBgXBXAtfMH10000EEEEEECUBYln03TTTdNBDZopopYvrTTdNa325mImNg3TTPV9q3pmY0xoO6bv3r00y+IDGid/9aaaZTGMuj9mpu9Mpio1dXrr5HERTZSmqU36A3CumzN/9Robv/Xx4v9ijkSRSNLQhAWumap82WRSBUqXStV/YcS+XVLnSS+WLDroqArFkMEsAS+eWmrUzrO0oEmE40RlMZ5+ODIkAyKAGUwZ3mVKmcamcJnMW26MRPgUw6j+LkhyHGVGYjSUUKNpuJUQoOIAyDvEyG8S5yfK6dhZc0Tx1KI/gviKL6qvvFs1+bWtaz58uUNnryq6kt5RzOCkPWlVqVX2a/EEBUdU1KrXLf40GoiiFXK///qpoiDXrOgqDR38JB0bw7SoL+ZB9o1RCkQjQ2CBYZKd/+VJxZRRZlqSkKiws0WFxUyCwsKiMy7hUVFhIaCrNQsKkTIsLivwKKigsj8XYlwt/WKi2N4d//uQRCSAAjURNIHpMZBGYiaQPSYyAAABLAAAAAAAACWAAAAApUF/Mg+0aohSIRobBAsMlO//Kk4soosy1JSFRYWaLC4qZBYWFRGZdwqKiwkNBVmoWFSJkWFxX4FFRQWR+LsS4W/rFRb/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////VEFHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU291bmRib3kuZGUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMjAwNGh0dHA6Ly93d3cuc291bmRib3kuZGUAAAAAAAAAACU=");



export class INDI extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      entries: {},
      messages: [],
      history: {}
    };
    
    if (this.props.history) {
      this.history_len = {};
      Object.keys(this.props.history).forEach(path => {
        var [device, name] = path.split(":");
        this.state.history[device] = this.state.history[device] || {};
        this.state.history[device][name] = this.state.history[device][name] || [];
        this.history_len[device] = this.history_len[device] || {};
        this.history_len[device][name] = this.props.history[path];
      });
    }

    
    [
      'actionSetProp', 'defSwitchVector', 'defTextVector', 'defNumberVector', 'defLightVector', 'defBLOBVector', 'setSwitchVector',
      'setTextVector', 'setNumberVector', 'setLightVector', 'setBLOBVector', 'delProperty', 'message', 'startWS'
    ].forEach(method => this[method] = this[method].bind(this));

    this.wsqueue = '';
    this.reconnect = false;
    this.wsauto = new Set(['<getProperties version="1.7"/>'])
    
    if (this.props.blob) {
      this.props.blob.forEach(path => {
        var [device, name] = path.split(":");
        this.enableBLOB(device, name);
      });
    }
  }

  componentDidMount() {
    this.reader = XmlReader.create({stream: true, emitTopLevelOnly: false, parentNodes: false});
    this.reader.on('tag:defSwitchVector', (data) => this.defSwitchVector(data));
    this.reader.on('tag:defTextVector', (data) => this.defTextVector(data));
    this.reader.on('tag:defNumberVector', (data) => this.defNumberVector(data));
    this.reader.on('tag:defLightVector', (data) => this.defLightVector(data));
    this.reader.on('tag:defBLOBVector', (data) => this.defBLOBVector(data));
    this.reader.on('tag:setSwitchVector', (data) => this.setSwitchVector(data));
    this.reader.on('tag:setNumberVector', (data) => this.setNumberVector(data));
    this.reader.on('tag:setTextVector', (data) => this.setTextVector(data));
    this.reader.on('tag:setLightVector', (data) => this.setLightVector(data));
    this.reader.on('tag:setBLOBVector', (data) => this.setBLOBVector(data));
    this.reader.on('tag:delProperty', (data) => this.delProperty(data));
    this.reader.on('tag:message', (data) => this.message(data));


    this.reconnect = true;
    this.startWS();
  }
  
  startWS() {
    if (this.webSocket !== undefined) return;

    var uri = this.props.wsurl;
    var loc = window.location;
    console.log(loc);
    if (loc.protocol === "https:") {
      uri = "wss:";
    } else {
      uri = "ws:";
    }
    if (loc.hostname === '127.0.0.1' || loc.hostname === 'localhost')
      uri += "//127.0.0.1:8080" + '/';
    else
      uri += "//" + loc.hostname + ':8081/';
    uri += 'websocket';

    console.log(uri);
    this.webSocket = new WebSocket(uri, ['binary', 'base64']);
    this.webSocket.onmessage = function (event) {
      if (event.data instanceof Blob) {
        var textreader = new FileReader();
        textreader.onload = function (event) {
          //alert(JSON.stringify(event.target.result, null, 4));
          this.reader.parse(event.target.result);
        }.bind(this);
        textreader.readAsText(event.data)
      }
      else {
        //alert(JSON.stringify(event, null, 4));
        this.reader.parse(event.data);
      }
    }.bind(this);
    this.webSocket.onopen = function (event) {
      this.reader.reset();
      this.reader.parse("<stream>");

      this.webSocket.send(Array.from(this.wsauto).join('') + this.wsqueue);
      this.wsqueue = '';
      this.setState({entries: {}});
    }.bind(this);
    this.webSocket.onerror = function (error) {
      this.webSocket.close();
      this.webSocket = undefined;
      console.log("WEBSOCKET", error);
    }.bind(this);
    this.webSocket.onclose = function (event) {
      this.webSocket = undefined;
      console.log("WEBSOCKET", event);
      if (this.reconnect) {
        beep_snd.play();
        setTimeout(this.startWS, 500);
      }
    }.bind(this);
  }

  componentWillUnmount() {
    this.reconnect = false;
    if (this.webSocket !== undefined) this.webSocket.close();
  }
  
  defSwitchVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'SwitchVector'}
      entry.elements = {};
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Off', i: i }));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  defTextVector(e) {
      //alert(JSON.stringify(e, null, 4));
      var entry = {...e.attributes, type: 'TextVector'}
      entry.elements = {};
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '', i: i}));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  defNumberVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'NumberVector'}
      entry.elements = {};
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '0', i: i}));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  defLightVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'LightVector'}
      entry.elements = {};
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Idle', i: i }));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  defBLOBVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'BLOBVector'}
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => {
          entry.elements = {};
          e.children.forEach((v, i) => {
            var value = '';
            try {
              value = prevState.entries[entry.device][entry.name].elements[v.attributes.name].value;
            }
            catch (error) {
              value = '';
            }
            entry.elements[v.attributes.name] = {...v.attributes, value: value, i: i }
          });
          return update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      });
      this.message(e);
  }

  setSwitchVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device] === undefined || this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Off' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  setNumberVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device] === undefined || this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '0' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  setTextVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device] === undefined || this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  setLightVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device] === undefined || this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Idle' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
  }

  setBLOBVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device] === undefined || this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? atob(v.children[0].value.trim()) : '' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
      this.updateHistory(entry.device, entry.name);
//      console.log("BLOB", this.state.entries[entry.device][entry.name]);
  }

  delProperty(e) {
      if ('name' in e.attributes) {
        this.setState(prevState => (
          update(prevState, {entries: {[e.attributes.device]: {$unset: [e.attributes.name]}}})
        ));
      }
      else {
        this.setState(prevState => (
          update(prevState, {entries: {$unset: [e.attributes.device]}})
        ));
      }
      this.message(e);
  }

  message(e) {
    
    if (e.attributes.state === 'Alert') beep_snd.play();
    if (!e.attributes.message) return;
//    this.setState(prevState => ({messages: [...prevState.messages.slice(-50), `${e.attributes.timestamp} ${e.attributes.device}: ${e.attributes.message}`]}));
    this.setState(prevState => ({messages: [`${e.attributes.timestamp} ${e.attributes.device}: ${e.attributes.message}`, ...prevState.messages.slice(0, 50)]}));
  }

  actionSetProp(device, name, changes) {
     if (!this.state.entries[device] || !this.state.entries[device][name]) {
       console.log(`prop ${device} ${name} does not exist`);
       return false;
     }
     var vector = this.state.entries[device][name];
     var vectorType = vector.type;
     var oneType = "one" + vectorType.slice(0, -6);
     vectorType = "new" + vectorType;

     if (vector.rule === 'OneOfMany' || vector.rule === 'AtMostOne') {
       var numOn = 0;
       var def_on;
       var en;
       for (en in changes) {
         console.log("changes1", en, changes[en]);
         if (changes[en] === 'On') {
           if (numOn === 0) numOn = 1;
           else changes[en] = 'Off'
         }
         def_on = en;
       }
       for (en in vector.elements) {
         var e = vector.elements[en];
         console.log("changes2_0", e);
         if (changes[e.name] !== undefined) continue;
         console.log("changes2", e.name, changes[e.name]);
         if (e.value === 'On') {
           if (numOn === 0) {
             numOn = 1;
             changes[e.name] = 'On';
           }
           else {
             changes[e.name] = 'Off';
           }
         }
         else {
           changes[e.name] = 'Off';
           if (numOn === 0) {
             def_on = e.name
           }
         }
       }
       if (vector.rule === 'OneOfMany' && numOn === 0) {
         changes[def_on] = 'On';
       }
     }
     
     console.log("Changes", changes);

     var elements_xml = '';
     Object.values(vector.elements).forEach(function (e) {
         var value = e.value;
         if (changes[e.name] !== undefined && (vector.type !== 'NumberVector' || changes[e.name] !== '')) {
           value = changes[e.name];
           elements_xml += '<' + oneType + ' name="' + e.name +'">' + value + '</' + oneType + '>';
         }
     });

     var xml = '<' + vectorType + ' device="' + device +'" name="' + name +'">' + elements_xml + '</' + vectorType + '>';
     console.log(xml);
     try {
         this.webSocket.send(xml);
     }
     catch (error) {
         console.log(error);
         this.wsqueue += xml;
     }
     return true;
  }

  enableBLOB(device, property) {
     var xml = `<enableBLOB device="${device}" name="${property}">Also</enableBLOB>`;
     console.log(xml);
     this.wsauto.add(xml);
     
     try {
         if (this.webSocket) this.webSocket.send(xml);
     }
     catch (error) {
         console.log(error);
     }
  }

  updateHistory(device, name) {
    if (!this.props.history ||
        !this.history_len[device] || !this.history_len[device][name] ||
        !this.state.history[device] || !this.state.history[device][name]) return;

    this.setState(prevState => {
      const entry = prevState.entries[device][name];
      //console.log(entry);
      const hist_entry = [];
      Object.values(entry.elements).forEach(e => {
        hist_entry[e.i] = e.value;
      });
      hist_entry.push(entry.timestamp);

      return update(prevState, {history: {[device]: {[name]: {$set:  prevState.history[device][name].concat([hist_entry]).slice(-this.history_len[device][name])   }}}})
    });

  }

  render() {
    return (
      <INDIContext.Provider value={ {state:this.state, indi:this} }>
        {this.props.children}
      </INDIContext.Provider>
    );
  }
}

export class INDIPanel extends React.Component {
  static contextType = INDIContext;

  constructor(props) {
    super(props);
    this.state = {
      tabIndex: 0 
    };
  }

  render() {
    var devices = {};
    Object.values(this.context.state.entries).forEach(dev => Object.values(dev).forEach(e => {
      if (!(e.device in devices)) devices[e.device] = {};
      if (!(e.group in devices[e.device])) devices[e.device][e.group] = {};
      devices[e.device][e.group][e.name] = e;
    }));
    
    
    return (
      <div id='INDI'>
        <Tabs forceRenderTabPanel selectedIndex={this.state.tabIndex} onSelect={tabIndex => this.setState({ tabIndex })}>
          <TabList>
            {Object.keys(devices).map(dev =>
              <Tab key={dev}>{dev}</Tab>
            )}
          </TabList>
          {Object.keys(devices).map(dev =>
            <TabPanel key={dev}>
              <INDIdevice 
                groups={devices[dev]} 
                indi={this.context.indi}
                key={dev}
                name={dev}
                extensions={React.Children.toArray(this.props.children)}
              />
            </TabPanel>
          )}
        </Tabs>
        <INDIMessages messages={this.context.state.messages}/>
      </div>
    );
  }
}
