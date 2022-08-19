pub mod format;

use std::collections::{HashMap, HashSet};

type Location = usize;
type State = usize;
type Name = usize;

struct Transitions {
  locations: usize,
  states: usize,
  transitions: HashMap<(State, Location), Vec<(Location, State)>>,
  // state -> accept?
  accept: Vec<bool>,
}

struct GadgetSpec {
  name: Name,
  // gadget location -> network location
  // mapping into a location is mandatory since otherwise
  // we would get bugs from not requiring transitivity
  locations: Vec<Location>,
}

struct Network {
  all_locations: usize,
  external_locations: usize,
  // note: names are not indices into gadgets or state-vectors!
  gadgets: Vec<GadgetSpec>,
  // named network state -> gadget state-vector
  states: Vec<Vec<State>>,
}

impl Network {
  // network location -> [(gadgetspec index, location)]
  fn reverse_location_map(&self) -> Vec<Vec<(usize, Location)>> {
    let mut result = vec![];
    result.resize(self.all_locations, vec![]);
    for (i, x) in self.gadgets.iter().enumerate().flat_map(|(n, g)| {
      g.locations
        .iter()
        .enumerate()
        .map(move |(k, &v)| (v, (n, k)))
    }) {
      result[i].push(x);
    }
    result
  }

  fn neighbors(
    &self,
    defs: &Vec<Transitions>,
    rlm: &Vec<Vec<(usize, Location)>>,
    states: Vec<State>,
    location: Location,
  ) -> Vec<(Location, Vec<State>)> {
    let mut result: Vec<(Location, Vec<State>)> = vec![];
    for &(gsi, gloc) in &rlm[location] {
      let gspec = &self.gadgets[gsi];
      let gadget = &defs[gspec.name];
      let gstate = states[gsi];
      if let Some(v) = gadget.transitions.get(&(gstate, gloc)) {
        for &(l, s) in v {
          let mut new_states = states.clone();
          new_states[gsi] = s;
          result.push((gspec.locations[l], new_states))
        }
      }
    }
    result
  }
}

/// Compute the state diagram of a Network
fn traverse_network(defs: Vec<Transitions>, net: Network) -> Transitions {
  let reverse_location_map = net.reverse_location_map();
  let transitions = HashMap::new();
  let all_external_states: HashSet<Vec<State>> = HashSet::new();

  todo!("compute transitions and external states");

  todo!("rename external states (consistently with net.states)");
  Transitions {
    locations: net.external_locations,
    states: all_external_states.len(),
    transitions,
    accept: todo!("compute accepting set of external states"),
  }
}
