

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

  if (pad === '') pad = ' ';

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
  var comp = s.match(/^ *(-?\d+(\.\d+)?) *((hr)|h|(deg)|º|:)? *(\d+(\.\d+)?)? *((min)|m|'|:)? *(\d+(\.\d+)?)? *((sec)|s|"|:)? *$/);
  if (comp == null) return '';
  
  var mul = 1.0;
  var res = 0;

  var num = parseFloat(comp[1]);
  res += mul * num;
  if (num < 0)
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
/*
  return (
    <button className={`INDIvector_item INDIvector_item_button ${props.name} ${props.vector_name} ${props.value === 'On' ? 'INDIvector_item_button_pressed' : 'INDIvector_item_button_released'}`} 
       onClick={props.onClick}
    >{props.label || props.name}</button>
  );
*/
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
      /> }
      {props.submit_show && <button
        className='INDIvector_item_set'
        onClick={props.onSubmit}
        disabled={props.submit_disabled}
      >set</button> }
    </span>
  );
}

class INDIproperty extends React.Component {

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
             onChange={() => this.props.actionSetProp(this.props.device, this.props.name, {[e.name]: (e.value === 'On'? "Off" : "On")})}
          />
        ))}
      </div>
    );
    if (this.props.type === 'LightVector') return (
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
    if (this.props.type === 'TextVector') return (
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
            onSubmit={() => this.props.actionSetProp(this.props.device, this.props.name, this.state.parsed)}
          />
          ))}
      </div>
    );
    if (this.props.type === 'NumberVector') {
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
            onSubmit={() => this.props.actionSetProp(this.props.device, this.props.name, this.state.parsed)}
          />
          ))}
      </div>
      );
    }
  }
}


function INDIgroup(props) {
  return (
    <div className={`INDIgroup ${props.name.replace(/\s+/g, '')}`}>
        <h3 className='INDIgroup_name'>{props.name}</h3>
        {Object.keys(props.vec).map(vec => React.createElement(INDIproperty, {...props.vec[vec], actionSetProp: props.actionSetProp, key: vec}, null)) }
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
              <INDIgroup vec={this.props.groups[group]} actionSetProp={this.props.actionSetProp} key={group} name={group}/>
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
          <div key={index}>{e}</div>
        ))}
      </div>
    </div>
  );
}


export default class INDI extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      entries: {},
      messages: [],
      tabIndex: 0 
    };
    this.actionSetProp = this.actionSetProp.bind(this);
    this.defSwitchVector = this.defSwitchVector.bind(this);
    this.defTextVector = this.defTextVector.bind(this);
    this.defNumberVector = this.defNumberVector.bind(this);
    this.defLightVector = this.defLightVector.bind(this);
    this.setSwitchVector = this.setSwitchVector.bind(this);
    this.setTextVector = this.setTextVector.bind(this);
    this.setNumberVector = this.setNumberVector.bind(this);
    this.setLightVector = this.setLightVector.bind(this);
    this.delProperty = this.delProperty.bind(this);
    this.message = this.message.bind(this);
    this.startWS = this.startWS.bind(this);

    this.wsqueue = '';
    this.reconnect = false;
  }

  componentDidMount() {
    this.reader = XmlReader.create({stream: true, emitTopLevelOnly: false, parentNodes: false});
    this.reader.on('tag:defSwitchVector', (data) => this.defSwitchVector(data));
    this.reader.on('tag:defTextVector', (data) => this.defTextVector(data));
    this.reader.on('tag:defNumberVector', (data) => this.defNumberVector(data));
    this.reader.on('tag:defLightVector', (data) => this.defLightVector(data));
    this.reader.on('tag:setSwitchVector', (data) => this.setSwitchVector(data));
    this.reader.on('tag:setNumberVector', (data) => this.setNumberVector(data));
    this.reader.on('tag:setTextVector', (data) => this.setTextVector(data));
    this.reader.on('tag:setLightVector', (data) => this.setLightVector(data));
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
    if (loc.hostname === '127.0.0.1')
      uri += "//" + loc.host + '/';
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

      this.webSocket.send('<getProperties version="1.7"/>' + this.wsqueue);
      this.wsqueue = '';
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
        setTimeout(this.startWS, 2000);
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
      entry.elements = {}
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Off', i: i }));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
  }

  defTextVector(e) {
      //alert(JSON.stringify(e, null, 4));
      var entry = {...e.attributes, type: 'TextVector'}
      entry.elements = {}
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '', i: i}));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
  }

  defNumberVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'NumberVector'}
      entry.elements = {}
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '0', i: i}));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
  }

  defLightVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes, type: 'LightVector'}
      entry.elements = {}
      e.children.forEach((v, i) => (entry.elements[v.attributes.name] = {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Idle', i: i }));
      //alert(JSON.stringify(entry, null, 4));
      this.setState(prevState => (
          update(prevState, {entries: { $auto: {[entry.device]: { $auto: {[entry.name]: { $set: entry }}}}}})
      ));
      this.message(e);
  }

  setSwitchVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Off' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
  }

  setNumberVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '0' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
  }

  setTextVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : '' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
  }

  setLightVector(e) {
      //console.log(e.children);
      var entry = {...e.attributes}
      if (this.state.entries[entry.device][entry.name] === undefined) {
        console.log("Undefined entry", e);
        return;
      }
      var elements = {};
      e.children.forEach(v => (elements[v.attributes.name] = {$merge: {...v.attributes, value: v.children.length ? v.children[0].value.trim() : 'Idle' }}));
      this.setState(prevState => (
          update(prevState, {entries: {[entry.device]: {[entry.name]: {$merge: entry, elements: elements }}}})
      ));
      this.message(e);
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
    if (!e.attributes.message) return;
//    this.setState(prevState => ({messages: [...prevState.messages.slice(-50), `${e.attributes.timestamp} ${e.attributes.device}: ${e.attributes.message}`]}));
    this.setState(prevState => ({messages: [`${e.attributes.timestamp} ${e.attributes.device}: ${e.attributes.message}`, ...prevState.messages.slice(0, 50)]}));
  }

  actionSetProp(device, name, changes) {
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
  }

  render() {
    var devices = {};
    Object.values(this.state.entries).forEach(dev => Object.values(dev).forEach(e => {
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
              <INDIdevice groups={devices[dev]} actionSetProp={this.actionSetProp} key={dev} name={dev} />
            </TabPanel>
          )}
        </Tabs>
        <INDIMessages messages={this.state.messages}/>
      </div>
    );
  }
}

