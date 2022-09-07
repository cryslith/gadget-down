use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Name(pub String);

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GadgetDescriptor {
  Transitions(Transitions),
  Network(Network),
  PostSelect(PostSelect),
}

#[derive(Debug, Deserialize)]
pub struct Gadget {
  #[serde(flatten)]
  g: GadgetDescriptor,
  #[serde(default)]
  determinize: bool,
  #[serde(default)]
  minimize: bool,
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
  #[serde(default, skip_serializing)]
  determinize: bool,
  #[serde(default, skip_serializing)]
  minimize: bool,
}

/// A gadget specified by postselecting another gadget
#[derive(Debug, Serialize, Deserialize)]
pub struct PostSelect {
  name: Name,
  underlying: Name,
  strict: bool,
  transitions: Vec<(Location, Location)>,
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
