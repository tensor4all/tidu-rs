#[allow(unused_imports)]
use chainrules::{ADKey, DiffPassId};

#[macro_export]
macro_rules! define_ad_key {
    ($name:ident) => {
        #[allow(dead_code)]
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            User(String),
            Tangent { of: Box<$name>, pass: DiffPassId },
        }

        impl ADKey for $name {
            fn tangent_of(&self, pass: DiffPassId) -> Self {
                $name::Tangent {
                    of: Box::new(self.clone()),
                    pass,
                }
            }
        }
    };
}
