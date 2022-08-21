pub mod format;

use std::collections::{HashMap, HashSet};

type Location = usize;
type State = usize;
type Name = usize;

struct Transitions {
  locations: usize,
  states: usize,
  transitions: HashMap<(Location, State), HashSet<(Location, State)>>,
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
        gadget.transitions.get(&(gloc, gstate)).map(move |v| {
          v.iter().map(move |&(l, s)| {
            let mut new_state = state.clone();
            new_state[gsi] = s;
            (gspec.locations[l], new_state)
          })
        })
      })
      .flatten()
  }

  /// Compute the state diagram of a Network
  fn transitions(&self, defs: Vec<Transitions>) -> Transitions {
    if self.external_locations == 0 {
      return Transitions {
        locations: 0,
        states: 0,
        transitions: HashMap::new(),
        accept: vec![],
      };
    }

    let rlm = self.reverse_location_map();

    // rename external states consistently with self.states
    let mut external_states: Vec<Vec<State>> = self.states.clone();
    let mut ext_states_inv: HashMap<Vec<State>, State> = self
      .states
      .iter()
      .enumerate()
      .map(|(i, s)| (s.clone(), i))
      .collect();

    let mut transitions: HashMap<(Location, State), HashSet<(Location, State)>> = HashMap::new();

    // search starting from every external state,
    // discovering new external states along the way
    let mut state_i = 0;
    while state_i < external_states.len() {
      let state = external_states[state_i].clone();
      for location in 0..self.external_locations {
        // search from (location, state)
        let mut frontier: Vec<(Location, Vec<State>)> = vec![(location, state.clone())];
        let mut seen: HashSet<(Location, Vec<State>)> = HashSet::new();
        let mut seen_external: HashSet<(Location, State)> = HashSet::new();
        while !frontier.is_empty() {
          let x = frontier.pop().unwrap();
          if seen.contains(&x) {
            continue;
          }
          seen.insert(x.clone());
          if x.0 < self.external_locations {
            let ind = if let Some(&ind) = ext_states_inv.get(&x.1) {
              ind
            } else {
              let ind = external_states.len();
              external_states.push(x.1.clone());
              ext_states_inv.insert(x.1.clone(), ind);
              ind
            };
            seen_external.insert((x.0, ind));
          }
          for n in self.neighbors(&defs, &rlm, x) {
            frontier.push(n);
          }
        }

        transitions.insert((location, state_i), seen_external);
      }
      state_i += 1;
    }

    let accept = external_states
      .iter()
      .map(|s| self.is_accepting(&defs, s))
      .collect();

    Transitions {
      locations: self.external_locations,
      states: external_states.len(),
      transitions,
      accept,
    }
  }

  fn is_accepting(&self, defs: &Vec<Transitions>, state: &Vec<State>) -> bool {
    state.iter().enumerate().all(|(i, &s)| {
      let gspec = &self.gadgets[i];
      let gadget = &defs[gspec.name];
      gadget.accept[s]
    })
  }
}
