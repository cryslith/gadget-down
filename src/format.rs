use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub type Location = String;
pub type State = String;
pub type Name = String;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Gadget {
  Transitions(Transitions),
  Network(Network),
}

/// A gadget specified as a state diagram
#[derive(Debug, Serialize, Deserialize)]
pub struct Transitions {
  name: Name,
  locations: Vec<Location>,
  states: Vec<State>,
  transitions: HashMap<State, Vec<(Location, Location, State)>>,
  /// A set of accepting states.  If none, defaults to all states.
  accept: Option<Vec<State>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GadgetSpec {
  name: Name,
  /// Mapping from locations of the named gadget to (internal or external) locations of the network.
  /// May omit locations.
  locations: HashMap<Location, Location>,
}

/// A gadget specified as a network of other named gadgets.
#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
  name: Name,
  locations: Vec<Location>,
  internal_locations: Vec<Location>,
  gadgets: Vec<GadgetSpec>,
  /// Not all possible states, just the ones important enough to name.
  /// Any states not reachable from these states are eliminated.
  /// State-vectors correspond to the gadgets field.
  states: HashMap<State, Vec<State>>,
}
