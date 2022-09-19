pub mod format;

use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use refinery::Partition;

type Location = usize;
type State = usize;
type Name = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transitions {
  pub locations: usize,
  pub states: usize,
  /// All trivial transitions (l, s) -> (l, s) are implicitly assumed to be present,
  /// unless some other transition (l, s1) -> (l, s2) is listed.
  /// If such a transition is listed then
  /// the trivial transition must be explicitly listed if it is present.
  pub transitions: HashMap<(Location, State), HashSet<(Location, State)>>,
  // state -> accept?
  pub accept: Vec<bool>,
}

// For compatibility with format
pub struct TranslateT {
  pub locations: HashMap<format::Location, Location>,
  pub states: HashMap<format::State, State>,
}

impl Transitions {
  /// Remove unnecessarily-listed trivial transitions from a (location, state) pair to itself,
  /// as well as empty transition lists.
  fn clean(&mut self) {
    self.transitions.retain(|&(l1, s1), v| {
      if !v.iter().any(|&(l2, s2)| l1 == l2 && s1 != s2) {
        v.remove(&(l1, s1));
      }
      !v.is_empty()
    });
  }

  /// Reflexively and transitively close the set of transitions.
  pub fn close(&mut self) {
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
    for &k in &heads {
      self.transitions.get_mut(&k).unwrap().insert(k);
    }
    self.clean();
  }

  pub fn is_deterministic(&self) -> bool {
    !self
      .transitions
      .iter()
      .any(|(_, s)| s.iter().map(|(l, _)| l).duplicates().any(|_| true))
  }

  pub fn determinize(&self) -> Self {
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

        // suppress trivial transitions since l1->l1 transition is unique
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

  // TODO p: &[State]
  fn renumber_states(&self, p: &[State]) -> Self {
    if p.len() != self.states {
      panic!("wrong number of states");
    }
    let transitions = self
      .transitions
      .iter()
      .map(|(&(l1, s1), v)| ((l1, p[s1]), v.iter().map(|&(l2, s2)| (l2, p[s2])).collect()))
      .collect();
    let mut accept = vec![false; self.states];
    for i in 0..self.states {
      accept[p[i]] = self.accept[i];
    }
    Self {
      locations: self.locations,
      states: self.states,
      transitions,
      accept,
    }
  }

  /// Minimize (and canonicalize) a deterministic gadget using Hopcroft's algorithm.
  /// The canonicalization approach is essentially Section 4.2
  /// from https://arxiv.org/abs/1306.2405
  pub fn minimize(&self) -> (Self, Vec<Option<State>>) {
    if !self.is_deterministic() {
      panic!("cannot minimize nondeterministic gadget");
    }

    let mut reverse_state_lookup: HashMap<State, HashMap<(Location, Location), HashSet<State>>> =
      HashMap::new();
    for (&(l1, s1), v) in &self.transitions {
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
      // don't need trivial transitions here since they can't
      // make a state not dead
      if let Some(m) = reverse_state_lookup.get(&x) {
        for (_, v) in m {
          for &s in v {
            frontier.push(s);
          }
        }
      }
    }

    if alive.is_empty() {
      return (
        Self {
          locations: self.locations,
          states: 1,
          transitions: HashMap::new(),
          accept: vec![false],
        },
        vec![Some(0); self.states],
      );
    }

    let (accept, reject): (Vec<usize>, Vec<usize>) = alive.iter().partition(|&&x| self.accept[x]);
    let mut partition = Partition::new(
      std::iter::once(accept.into_iter()).chain(if reject.is_empty() {
        None
      } else {
        Some(reject.into_iter())
      }),
      self.states,
    );

    // set of parts which are needed for further splits
    // maintained as a vec for determinism
    let mut distinguishers: Vec<usize> = (0..partition.num_parts()).collect();
    let mut distinguishers_set: HashSet<usize> = (0..partition.num_parts()).collect();
    // maintain a deterministic order of parts in the partition
    let mut parts_order: Vec<usize> = (0..partition.num_parts()).collect();
    while !distinguishers.is_empty() {
      let a = distinguishers.pop().unwrap();
      distinguishers_set.remove(&a);
      let mut transitions_into_a: HashMap<(Location, Location), HashSet<State>> = HashMap::new();
      for &x in partition.part(a) {
        if let Some(m) = reverse_state_lookup.get(&x) {
          for (&k, v) in m.iter() {
            transitions_into_a.entry(k).or_default().extend(v);
          }
        }
        // add trivial transitions
        for l in 0..self.locations {
          if self
            .transitions
            .get(&(l, x))
            .map(|v| !v.iter().any(|&(l2, _)| l == l2))
            .unwrap_or(true)
          {
            transitions_into_a.entry((l, l)).or_default().insert(x);
          }
        }
      }

      // sort by input character
      for ((l1, l2), v) in transitions_into_a.into_iter().sorted_by_key(|&(k, _)| k) {
        let v: Vec<State> = v.into_iter().collect();
        let mut new_parts = vec![];
        partition.refine_with_callback(&v[..], |partition, orig, new| {
          new_parts.push((orig, new));
        });
        // sort the new parts by old part number
        new_parts.sort_by_cached_key(|&(old, _)| parts_order[old]);
        for &(orig, new) in &new_parts {
          if distinguishers_set.contains(&orig) {
            // both orig and new are needed since orig was needed
            distinguishers.push(new);
            distinguishers_set.insert(new);
          } else {
            // orig wasn't needed so we only need one of {orig, new}
            // we choose orig instead of the smaller one to avoid
            // diverging behavior for equivalent gadgets
            distinguishers.push(orig);
            distinguishers_set.insert(orig);
          }
        }

        let n = parts_order.len();
        parts_order.resize(n + new_parts.len(), 0);
        for (i, (_, new)) in new_parts.into_iter().enumerate() {
          parts_order[new] = n + i;
        }
      }
    }

    let mut old_state_mapping = vec![None; self.states];
    for s in alive {
      old_state_mapping[s] = Some(partition.find(s));
    }

    let mut transitions: HashMap<(Location, State), HashSet<(Location, State)>> = HashMap::new();
    for (&(l1, s1), v) in &self.transitions {
      let part1 = if let Some(p) = old_state_mapping[s1] {
        parts_order[p]
      } else {
        continue;
      };
      if transitions.contains_key(&(l1, part1)) {
        continue;
      }
      let mut v_new = HashSet::new();
      for &(l2, s2) in v {
        if let Some(p) = old_state_mapping[s2] {
          v_new.insert((l2, parts_order[p]));
        }
      }

      // suppress trivial transition since gadget is deterministic
      v_new.remove(&(l1, part1));

      if !v_new.is_empty() {
        transitions.insert((l1, part1), v_new);
      }
    }

    let mut accept = vec![false; partition.num_parts()];
    for (s, &x) in old_state_mapping.iter().enumerate() {
      if let Some(p) = x {
        accept[parts_order[p]] = self.accept[s];
      }
    }

    (
      Self {
        locations: self.locations,
        states: partition.num_parts(),
        transitions,
        accept,
      },
      old_state_mapping,
    )
  }

  fn from_format(t: format::Transitions) -> (Self, TranslateT) {
    let nstates = t.states.len();
    let nloc = t.locations.len();
    let state_map: HashMap<format::State, State> = t
      .states
      .into_iter()
      .enumerate()
      .map(|(i, x)| (x, i))
      .collect();
    let location_map: HashMap<format::Location, Location> = t
      .locations
      .into_iter()
      .enumerate()
      .map(|(i, x)| (x, i))
      .collect();
    let mut transitions: HashMap<(Location, State), HashSet<(Location, State)>> = HashMap::new();
    for (s1, v) in t.transitions {
      let s1 = state_map[&s1];
      for (l1, l2, s2) in v {
        let l1 = location_map[&l1];
        let l2 = location_map[&l2];
        let s2 = state_map[&s2];
        transitions.entry((l1, s1)).or_default().insert((l2, s2));
      }
    }
    let mut accept = vec![false; nstates];
    if let Some(v) = t.accept {
      for s in v {
        accept[state_map[&s]] = true;
      }
    }
    (
      Self {
        locations: nloc,
        states: nstates,
        transitions,
        accept,
      },
      TranslateT {
        locations: location_map,
        states: state_map,
      },
    )
  }
}

impl From<format::Transitions> for Transitions {
  fn from(t: format::Transitions) -> Self {
    Self::from_format(t).0
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
    defs: &'a [Transitions],
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
  /// The result is reflexively and transitively closed.
  // TODO skip listed trivial transitions
  fn transitions(&self, defs: &[Transitions]) -> Transitions {
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

    let mut result = Transitions {
      locations: self.external_locations,
      states: external_states.len(),
      transitions,
      accept,
    };
    result.clean();
    result
  }

  fn is_accepting(&self, defs: &[Transitions], state: &Vec<State>) -> bool {
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
      accept: vec![true; 2],
    }
  }

  fn choice_crumbler() -> Transitions {
    Transitions {
      locations: 3,
      states: 2,
      transitions: [((0, 0), [(1, 1), (2, 1)].into_iter().collect())]
        .into_iter()
        .collect(),
      accept: vec![true; 2],
    }
  }

  #[test]
  fn minimize_minimized() {
    for x in [diode(), l2t(), choice_crumbler()] {
      assert_eq!(x, x.minimize().0);
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
    assert_eq!(
      t,
      Transitions {
        locations: 2,
        states: 2,
        transitions: [
          ((0, 0), [(1, 1)].into_iter().collect()),
          ((1, 1), [(0, 0)].into_iter().collect()),
        ]
        .into_iter()
        .collect(),

        accept: vec![true; 2],
      }
    );
    assert!(t.is_deterministic());
    assert_eq!(t, t.minimize().0);
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
    assert_eq!(
      t,
      Transitions {
        locations: 3,
        states: 4,
        transitions: [
          ((0, 0), [(1, 1)].into_iter().collect()),
          ((2, 0), [(2, 0), (2, 2), (2, 3)].into_iter().collect()),
          ((1, 1), [(0, 0)].into_iter().collect()),
          ((2, 1), [(2, 1), (2, 2), (2, 3)].into_iter().collect()),
          ((0, 2), [(1, 3)].into_iter().collect()),
          ((1, 3), [(0, 2)].into_iter().collect()),
        ]
        .into_iter()
        .collect(),
        accept: vec![true; 4]
      }
    );
    assert!(!t.is_deterministic());

    let t2 = t.determinize();
    let u2 = Transitions {
      locations: 3,
      states: 8,
      transitions: [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((2, 0), [(2, 4)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
        ((2, 1), [(2, 5)].into_iter().collect()),
        ((0, 2), [(1, 3)].into_iter().collect()),
        ((1, 3), [(0, 2)].into_iter().collect()),
        ((0, 4), [(1, 6)].into_iter().collect()),
        ((1, 4), [(0, 2)].into_iter().collect()),
        ((0, 5), [(1, 3)].into_iter().collect()),
        ((1, 5), [(0, 7)].into_iter().collect()),
        ((1, 6), [(0, 7)].into_iter().collect()),
        ((2, 6), [(2, 5)].into_iter().collect()),
        ((0, 7), [(1, 6)].into_iter().collect()),
        ((2, 7), [(2, 4)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
      accept: vec![true; 8],
    }
    .renumber_states(&[0, 1, 2, 3, 7, 4, 6, 5]);
    // for x in [&t2, &u2] {
    //   println!("{:?}", x.transitions.iter().sorted_by_key(|(&(l1, s1), _)| (s1, l1)));
    // }
    assert_eq!(t2, u2);

    let t3 = t2.minimize().0;
    let u3 = Transitions {
      locations: 3,
      states: 6,
      transitions: [
        ((0, 0), [(1, 1)].into_iter().collect()),
        ((2, 0), [(2, 4)].into_iter().collect()),
        ((1, 1), [(0, 0)].into_iter().collect()),
        ((2, 1), [(2, 5)].into_iter().collect()),
        ((0, 2), [(1, 3)].into_iter().collect()),
        ((1, 3), [(0, 2)].into_iter().collect()),
        ((0, 4), [(1, 1)].into_iter().collect()),
        ((1, 4), [(0, 2)].into_iter().collect()),
        ((0, 5), [(1, 3)].into_iter().collect()),
        ((1, 5), [(0, 0)].into_iter().collect()),
      ]
      .into_iter()
      .collect(),
      accept: vec![true; 6],
    }
    .renumber_states(&[3, 5, 2, 1, 0, 4]);
    for x in [&t3, &u3, &t3.minimize().0] {
      println!(
        "{:?}",
        x.transitions
          .iter()
          .sorted_by_key(|(&(l1, s1), _)| (s1, l1))
      );
    }
    assert_eq!(t3, u3);
    assert_eq!(t3, t3.minimize().0);
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
      accept: vec![true; 2],
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

    let (t3, m) = t2.minimize();
    assert_eq!(t3, otc_door());
    assert_eq!(m, vec![Some(0), Some(1), Some(0)]);
  }

  #[test]
  fn test_minimize_trivial_transition() {
    let t = Transitions {
      locations: 1,
      states: 2,
      transitions: [((0, 1), [(0, 0)].into_iter().collect())]
        .into_iter()
        .collect(),
      accept: vec![true, true],
    };
    let u = Transitions {
      locations: 1,
      states: 1,
      transitions: HashMap::new(),
      accept: vec![true],
    };
    assert_eq!(t.minimize().0, u);
  }
}
