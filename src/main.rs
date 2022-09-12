use gadget_down::*;
use format::GadgetDescriptor;

fn main() {
  let inputs: Vec<format::Gadget> = serde_yaml::from_str(include_str!("../example-input.yaml")).unwrap();
  let outputs: Vec<Transitions> = vec![];
  for (i, format::Gadget { g, determinize, minimize }) in inputs.into_iter().enumerate() {
    let t: Transitions = match g {
      GadgetDescriptor::Transitions(t) => {
        t.into()
      }
      GadgetDescriptor::Network(n) => {
        todo!()
      }
      GadgetDescriptor::PostSelect(n) => {
        todo!()
      }
    };
  }
}
