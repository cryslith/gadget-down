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

impl Transitions {
  /// Remove trivial locations from a (location, state) pair to itself.
  fn remove_trivial(&mut self) {
    for (k, v) in self.transitions.iter_mut() {
      v.remove(k);
    }
  }

  /// Transitively close the set of transitions.
  fn transitive_close(&mut self) {
    let heads: Vec<(Location, State)> = self.transitions.keys().cloned().collect();
    let tails: Vec<(Location, State)> = self
      .transitions
      .values()
      .map(|s| s.iter())
      .flatten()
      .cloned()
      .collect();
    for k in &heads {
      for i in &heads {
        for j in &tails {
          if self.transitions[i].contains(k) && self.transitions[k].contains(j) {
            self.transitions.get_mut(i).unwrap().insert(*j);
          }
        }
      }
    }
  }
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

  /// Compute the state diagram of a Network.
  /// The result has no trivial transitions and is transitively closed.
  fn transitions(&self, defs: &Vec<Transitions>) -> Transitions {
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
          for n in self.neighbors(defs, &rlm, x) {
            frontier.push(n);
          }
        }

        // suppress trivial transition
        seen_external.remove(&(location, state_i));

        if !seen_external.is_empty() {
          transitions.insert((location, state_i), seen_external);
        }
      }
      state_i += 1;
    }

    let accept = external_states
      .iter()
      .map(|s| self.is_accepting(defs, s))
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

#[cfg(test)]
mod tests {
  use super::*;

  fn diode() -> Transitions {
    Transitions {
      locations: 2,
      states: 1,
      transitions: [((0, 0), [(1, 0)].into_iter().collect())]
        .into_iter()
        .collect(),
      accept: vec![true],
    }
  }

  fn l2t() -> Transitions {
    Transitions {
      locations: 4,
      states: 2,
      transitions: [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((2, 0), [(3, 1)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
        ((3, 1), [(2, 0)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
      accept: vec![true, true],
    }
  }

  fn choice_crumbler() -> Transitions {
    Transitions {
      locations: 3,
      states: 2,
      transitions: [((0, 0), [(1, 1), (2, 1)].into_iter().collect())]
        .into_iter()
        .collect(),
      accept: vec![true, true],
    }
  }

  fn network_1() -> (Vec<Transitions>, Network) {
    (
      vec![diode(), l2t(), choice_crumbler()],
      Network {
        all_locations: 7,
        external_locations: 2,
        gadgets: vec![
          GadgetSpec {
            name: 1,
            locations: vec![0, 2, 3, 4],
          },
          GadgetSpec {
            name: 1,
            locations: vec![1, 2, 5, 6],
          },
        ],
        states: vec![vec![0, 1]],
      },
    )
  }

  #[test]
  fn solve_network_1() {
    let (defs, n) = network_1();
    let t = n.transitions(&defs);
    assert_eq!(t.locations, 2);
    assert_eq!(t.states, 2);
    assert_eq!(t.accept, vec![true, true]);
    assert_eq!(
      t.transitions,
      [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
    );
  }

  fn network_2() -> (Vec<Transitions>, Network) {
    (
      vec![diode(), l2t(), choice_crumbler()],
      Network {
        all_locations: 5,
        external_locations: 3,
        gadgets: vec![
          GadgetSpec {
            name: 1,
            locations: vec![0, 1, 3, 4],
          },
          GadgetSpec {
            name: 2,
            locations: vec![2, 3, 4],
          },
          GadgetSpec {
            name: 0,
            locations: vec![3, 2],
          },
          GadgetSpec {
            name: 0,
            locations: vec![4, 2],
          },
        ],
        states: vec![
          vec![0, 0, 0, 0],
          vec![1, 0, 0, 0],
          vec![0, 1, 0, 0],
          vec![1, 1, 0, 0],
        ],
      },
    )
  }

  #[test]
  fn solve_network_2() {
    let (defs, n) = network_2();
    let t = n.transitions(&defs);
    assert_eq!(t.locations, 3);
    assert_eq!(t.states, 4);
    assert_eq!(t.accept, vec![true, true, true, true]);
    assert_eq!(
      t.transitions,
      [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
        ((0, 2), [(1, 3)].into_iter().collect()),
        ((1, 3), [(0, 2)].into_iter().collect()),
        ((2, 0), [(2, 2), (2, 3)].into_iter().collect()),
        ((2, 1), [(2, 2), (2, 3)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
    );
  }

  fn otc_door() -> Transitions {
    Transitions {
      locations: 6,
      states: 2,
      transitions: [
        ((0, 0), [(1, 0)].into_iter().collect()),
        ((2, 0), [(3, 0)].into_iter().collect()),
        ((4, 0), [(5, 1)].into_iter().collect()),
        ((0, 1), [(1, 0)].into_iter().collect()),
        ((4, 1), [(5, 1)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
      accept: vec![true, true],
    }
  }

  fn network_3() -> (Vec<Transitions>, Network) {
    (
      vec![otc_door()],
      Network {
        all_locations: 15,
        external_locations: 6,
        gadgets: vec![
          GadgetSpec {
            name: 0,
            locations: vec![6, 6, 2, 3, 4, 5],
          },
          GadgetSpec {
            name: 0,
            locations: vec![7, 8, 0, 6, 9, 10],
          },
          GadgetSpec {
            name: 0,
            locations: vec![11, 12, 6, 1, 13, 14],
          },
        ],
        states: vec![vec![0, 0, 0], vec![1, 0, 0]],
      },
    )
  }

  #[test]
  fn solve_network_3() {
    let (defs, n) = network_3();
    let t = n.transitions(&defs);
    assert_eq!(t.locations, 6);
    assert_eq!(t.states, 2);
    assert_eq!(t.accept, vec![true, true]);
    let mut transitions = otc_door().transitions;
    transitions.get_mut(&(0, 1)).unwrap().insert((1, 1));
    assert_eq!(t.transitions, transitions);
  }
}
