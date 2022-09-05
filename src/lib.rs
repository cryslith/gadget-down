pub mod format;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use refinery::Partition;

type Location = usize;
type State = usize;
type Name = usize;

#[derive(Debug, Clone)]
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

  fn is_deterministic(&self) -> bool {
    !self
      .transitions
      .iter()
      .any(|(_, s)| s.iter().map(|(l, _)| l).duplicates().any(|_| true))
  }

  fn determinize(&self) -> Self {
    if self.is_deterministic() {
      return self.clone();
    }
    if self.states > 64 {
      // TODO use a trait for state sets to switch between u64 or hashset
      panic!("too many states");
    }

    fn from_states(x: impl IntoIterator<Item = State>) -> u64 {
      x.into_iter().fold(0, |a, b| a | (1 << b))
    }

    fn to_states(mut x: u64) -> impl Iterator<Item = State> {
      let mut i = 0;
      std::iter::from_fn(move || {
        if x == 0 {
          None
        } else {
          while x & 1 == 0 {
            x = x >> 1;
            i += 1;
          }
          x = x >> 1;
          i += 1;
          Some(i - 1)
        }
      })
    }

    let mut new_states: Vec<u64> = (0..self.states).map(|s| 1 << s).collect();
    let mut new_states_inv: HashMap<u64, usize> = new_states
      .iter()
      .enumerate()
      .map(|(i, &x)| (x, i))
      .collect();
    let mut transitions = HashMap::new();
    let mut seen = HashSet::new();
    let mut frontier: Vec<State> = (0..new_states.len()).collect();
    while !frontier.is_empty() {
      let x = frontier.pop().unwrap();
      if seen.contains(&x) {
        continue;
      }
      seen.insert(x);
      for l1 in 0..self.locations {
        let l1x_transitions = to_states(new_states[x])
          .filter_map(|s| self.transitions.get(&(l1, s)).map(|g| g.iter()))
          .flatten()
          .cloned()
          .into_group_map();
        let mut l1x_new_transitions = HashSet::new();
        for (l2, states) in l1x_transitions {
          let y_s = from_states(states);
          let y_i = if let Some(&y_i) = new_states_inv.get(&y_s) {
            y_i
          } else {
            let y_i = new_states.len();
            new_states.push(y_s);
            new_states_inv.insert(y_s, y_i);
            y_i
          };
          l1x_new_transitions.insert((l2, y_i));
          frontier.push(y_i);
        }

        l1x_new_transitions.remove(&(l1, x));
        if !l1x_new_transitions.is_empty() {
          transitions.insert((l1, x), l1x_new_transitions);
        }
      }
    }
    let accept = new_states
      .iter()
      .map(|&x_s| to_states(x_s).any(|s| self.accept[s]))
      .collect();
    Self {
      locations: self.locations,
      states: new_states.len(),
      transitions,
      accept,
    }
  }

  /// Minimize a deterministic gadget using Hopcroft's algorithm
  fn minimize(&self) -> Self {
    if !self.is_deterministic() {
      panic!("cannot minimize nondeterministic gadget");
    }
    let mut reverse_state_lookup: HashMap<State, HashMap<(Location, Location), HashSet<State>>> =
      HashMap::new();
    for (&(s1, l1), v) in &self.transitions {
      for &(l2, s2) in v {
        reverse_state_lookup
          .entry(s2)
          .or_default()
          .entry((l1, l2))
          .or_default()
          .insert(s1);
      }
    }

    // ignore dead states
    let mut alive: HashSet<usize> = HashSet::new();
    let mut frontier: Vec<State> = (0..self.states).filter(|&x| self.accept[x]).collect();
    while !frontier.is_empty() {
      let x = frontier.pop().unwrap();
      if alive.contains(&x) {
        continue;
      }
      alive.insert(x);
      if let Some(m) = reverse_state_lookup.get(&x) {
        for (_, v) in m {
          for &s in v {
            frontier.push(s);
          }
        }
      }
    }

    let (accept, reject): (Vec<usize>, Vec<usize>) = alive.iter().partition(|&&x| self.accept[x]);
    let mut partition = Partition::new(
      [accept.into_iter(), reject.into_iter()].into_iter(),
      self.states + 1,
    );

    // set of parts which are needed for further splits
    let mut distinguishers: HashSet<usize> = [0, 1].into_iter().collect();
    while !distinguishers.is_empty() {
      let a = *distinguishers.iter().next().unwrap();
      distinguishers.remove(&a);
      let mut transitions_into_a: HashMap<(Location, Location), HashSet<State>> = HashMap::new();
      for x in partition.part(a) {
        if let Some(m) = reverse_state_lookup.get(x) {
          for (k, v) in m.iter() {
            transitions_into_a.entry(k).or_default().extend(v);
          }
        }
      }

      for (k, v) in transitions_into_a {
        let v: Vec<State> = v.iter().cloned().collect();
        partition.refine_with_callback(&v[..], |partition, orig, new| {
          if distinguishers.contains(&orig) {
            // both orig and new are needed since orig was needed
            distinguishers.insert(new);
          } else {
            // orig wasn't needed so we only need one of {orig, new}
            let smaller = if partition.part(new).len() < partition.part(orig).len() {
              new
            } else {
              orig
            };
            distinguishers.insert(smaller);
          }
        })
      }
    }

    let transitions = todo!();
    let accept = todo!();
    Self {
      locations: self.locations,
      states: partition.num_parts(),
      transitions,
      accept,
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
    assert!(t.is_deterministic());
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
    assert!(!t.is_deterministic());

    let t2 = t.determinize();
    assert_eq!(t2.locations, 3);
    assert_eq!(t2.states, 5);
    assert_eq!(t2.accept, vec![true, true, true, true, true]);
    assert_eq!(
      t2.transitions,
      [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
        ((0, 2), [(1, 3)].into_iter().collect()),
        ((1, 3), [(0, 2)].into_iter().collect()),
        ((2, 0), [(2, 4)].into_iter().collect()),
        ((2, 1), [(2, 4)].into_iter().collect()),
        ((0, 4), [(1, 3)].into_iter().collect()),
        ((1, 4), [(0, 2)].into_iter().collect()),
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
    assert!(!t.is_deterministic());

    let t2 = t.determinize();
    assert_eq!(t2.locations, 6);
    assert_eq!(t2.states, 3);
    assert_eq!(t2.accept, vec![true, true, true]);
    assert_eq!(
      t2.transitions,
      [
        ((0, 0), [(1, 0)].into_iter().collect()),
        ((2, 0), [(3, 0)].into_iter().collect()),
        ((4, 0), [(5, 1)].into_iter().collect()),
        ((0, 1), [(1, 2)].into_iter().collect()),
        ((4, 1), [(5, 1)].into_iter().collect()),
        ((0, 2), [(1, 2)].into_iter().collect()),
        ((2, 2), [(3, 0)].into_iter().collect()),
        ((4, 2), [(5, 1)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
    );
  }
}
