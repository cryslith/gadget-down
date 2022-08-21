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

  fn external_states(&self, all_nodes: &HashSet<(Location, Vec<State>)>) -> HashSet<Vec<State>> {
    all_nodes
      .iter()
      .filter(|(l, s)| *l < self.external_locations)
      .map(|(l, s)| s)
      .cloned()
      .collect()
  }

  fn reachability(
    &self,
    defs: &Vec<Transitions>,
    rlm: &Vec<Vec<(usize, Location)>>,
    all_nodes: &HashSet<(Location, Vec<State>)>,
    external_states: &HashSet<Vec<State>>,
  ) -> HashMap<(Location, Vec<State>), HashSet<(Location, Vec<State>)>> {
    let mut result: HashMap<(Location, Vec<State>), HashSet<(Location, Vec<State>)>> =
      HashMap::new();
    for i in all_nodes {
      result.insert(i.clone(), self.neighbors(defs, rlm, i.clone()).collect());
    }
    for k in all_nodes {
      for i in all_nodes {
        for j in all_nodes {
          if result[i].contains(k) && result[k].contains(j) {
            result.get_mut(i).unwrap().insert(j.clone());
          }
        }
      }
    }
    result.retain(|(l, s), _| external_states.contains(s));
    result
  }

  fn is_accepting(&self, defs: &Vec<Transitions>, state: &Vec<State>) -> bool {
    state.iter().enumerate().all(|(i, &s)| {
      let gspec = &self.gadgets[i];
      let gadget = &defs[gspec.name];
      gadget.accept[s]
    })
  }

  /// Compute the state diagram of a Network
  fn transitions(&self, defs: Vec<Transitions>) -> Transitions {
    let rlm = self.reverse_location_map();
    let all_nodes = self.all_nodes(&defs, &rlm);
    let mut external_states = self.external_states(&all_nodes);
    let n_external_states = external_states.len();
    let reachability = self.reachability(&defs, &rlm, &all_nodes, &external_states);

    // rename external states consistently with net.states
    let mut state_renaming: Vec<Vec<State>> = self.states.clone();
    for s in &self.states {
      external_states.remove(s);
    }
    for s in external_states {
      state_renaming.push(s);
    }
    let state_renaming_inv: HashMap<Vec<State>, State> = state_renaming
      .iter()
      .enumerate()
      .map(|(i, s)| (s.clone(), i))
      .collect();

    let transitions: HashMap<(Location, State), HashSet<(Location, State)>> = reachability
      .iter()
      .map(|((l, s), n)| {
        (
          (*l, state_renaming_inv[s]),
          n.iter()
            .map(|(l2, s2)| (*l2, state_renaming_inv[s2]))
            .collect(),
        )
      })
      .collect();

    let accept = state_renaming
      .iter()
      .map(|s| self.is_accepting(&defs, s))
      .collect();

    Transitions {
      locations: self.external_locations,
      states: n_external_states,
      transitions,
      accept,
    }
  }
}
