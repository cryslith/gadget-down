use gadget_down::*;

fn main() {
  let x: Vec<format::Gadget> = serde_yaml::from_str(include_str!("../example-input.yaml")).unwrap();
  println!("{:?}", x);
}
