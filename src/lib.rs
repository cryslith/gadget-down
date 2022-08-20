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

  fn neighbors<'a>(
    &'a self,
    defs: &'a Vec<Transitions>,
    rlm: &'a Vec<Vec<(usize, Location)>>,
    node: (Location, Vec<State>),
  ) -> impl Iterator<Item = (Location, Vec<State>)> + 'a {
    let (location, state) = node;
    rlm[location]
      .iter()
      .filter_map(move |&(gsi, gloc)| {
        let state = state.clone();
        let gspec = &self.gadgets[gsi];
        let gadget = &defs[gspec.name];
        let gstate = state[gsi];
        gadget.transitions.get(&(gstate, gloc)).map(move |v| {
          v.iter().map(move |&(l, s)| {
            let mut new_state = state.clone();
            new_state[gsi] = s;
            (gspec.locations[l], new_state)
          })
        })
      })
      .flatten()
  }

  fn neighbors_external<'a>(
    &'a self,
    defs: &'a Vec<Transitions>,
    rlm: &'a Vec<Vec<(usize, Location)>>,
    node: (Location, Vec<State>),
  ) -> impl Iterator<Item = (Location, Vec<State>)> + 'a {
    let (location, state) = node;
    let state_ = state.clone();
    self.neighbors(defs, rlm, (location, state)).chain(
      if location < self.external_locations {
        Some((0..self.external_locations).filter_map(move |l| {
          if l == location {
            None
          } else {
            Some((l, state_.clone()))
          }
        }))
      } else {
        None
      }
      .into_iter()
      .flatten(),
    )
  }

  fn all_nodes(
    &self,
    defs: &Vec<Transitions>,
    rlm: &Vec<Vec<(usize, Location)>>,
  ) -> HashSet<(Location, Vec<State>)> {
    let mut result = HashSet::new();
    if self.external_locations == 0 {
      return result;
    }
    let mut frontier: Vec<(Location, Vec<State>)> = self
      .states
      .iter()
      .map(|s| (0, s.clone())) // external location 0 is arbitrary
      .collect();
    while !frontier.is_empty() {
      let x = frontier.pop().unwrap();
      if result.contains(&x) {
        continue;
      }
      result.insert(x.clone());
      for n in self.neighbors_external(defs, rlm, x) {
        frontier.push(n);
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
