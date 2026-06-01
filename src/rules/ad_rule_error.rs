use std::error::Error;
use std::fmt;

/// Identifies which AD rule failed or is unavailable.
///
/// # Examples
///
/// ```
/// use tidu::ADRuleKind;
///
/// assert_eq!(ADRuleKind::Jvp.as_str(), "jvp");
/// assert_eq!(ADRuleKind::Transpose.as_str(), "transpose");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ADRuleKind {
    /// JVP rule for forward linearization.
    Jvp,
    /// Transpose / VJP rule for a linear primitive.
    Transpose,
}

impl ADRuleKind {
    /// Returns a stable human-readable rule name.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::ADRuleKind;
    ///
    /// assert_eq!(ADRuleKind::Jvp.as_str(), "jvp");
    /// ```
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Jvp => "jvp",
            Self::Transpose => "transpose",
        }
    }
}

/// Error returned when an AD rule cannot be emitted.
///
/// # Examples
///
/// ```
/// use tidu::{ADRuleError, ADRuleKind};
///
/// let err = ADRuleError::unsupported("my_crate::fft", ADRuleKind::Jvp);
/// assert_eq!(err.rule(), ADRuleKind::Jvp);
/// assert!(err.to_string().contains("my_crate::fft"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ADRuleError {
    /// The requested primitive does not provide the requested AD rule.
    Unsupported {
        /// Stable primitive name or extension family identifier.
        op: String,
        /// Missing rule kind.
        rule: ADRuleKind,
    },
}

impl ADRuleError {
    /// Constructs an unsupported-rule error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{ADRuleError, ADRuleKind};
    ///
    /// let err = ADRuleError::unsupported("custom::op", ADRuleKind::Transpose);
    /// assert_eq!(err.rule(), ADRuleKind::Transpose);
    /// ```
    pub fn unsupported(op: impl Into<String>, rule: ADRuleKind) -> Self {
        Self::Unsupported {
            op: op.into(),
            rule,
        }
    }

    /// Returns the AD rule kind associated with this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use tidu::{ADRuleError, ADRuleKind};
    ///
    /// let err = ADRuleError::unsupported("custom::op", ADRuleKind::Jvp);
    /// assert_eq!(err.rule(), ADRuleKind::Jvp);
    /// ```
    #[cfg_attr(coverage, inline(never))]
    pub const fn rule(&self) -> ADRuleKind {
        match self {
            Self::Unsupported { rule, .. } => *rule,
        }
    }
}

impl fmt::Display for ADRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { op, rule } => {
                write!(f, "unsupported {} AD rule for {op}", rule.as_str())
            }
        }
    }
}

impl Error for ADRuleError {}

/// Result type used by fallible AD rule emission.
///
/// # Examples
///
/// ```
/// use tidu::{ADRuleError, ADRuleKind, ADRuleResult};
///
/// fn missing_rule() -> ADRuleResult<()> {
///     Err(ADRuleError::unsupported("custom::op", ADRuleKind::Transpose))
/// }
///
/// assert!(missing_rule().is_err());
/// ```
pub type ADRuleResult<T> = Result<T, ADRuleError>;
