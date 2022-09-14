use gadget_down::*;
use format::GadgetDescriptor;

fn main() {
  let inputs: Vec<format::Gadget> = serde_yaml::from_str(include_str!("../example-input.yaml")).unwrap();
  let mut outputs: Vec<Transitions> = vec![];
  for (i, format::Gadget { g, determinize, minimize }) in inputs.into_iter().enumerate() {
    let mut t: Transitions = match g {
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
    t.close();
    if determinize || minimize {
      t = t.determinize();
    }
    if minimize {
      t = t.minimize().0;
    }
    outputs.push(t);
  }
}
