use std::iter::{FromIterator, once};
use std::ops::{Deref, DerefMut};
use std::collections::HashMap;

use glsl::syntax::{NonEmpty, TypeQualifier, TypeSpecifier, TypeSpecifierNonArray};
use glsl::syntax;
use glsl::syntax::StructFieldSpecifier;
use glsl::syntax::PrecisionQualifier;
use glsl::syntax::ArrayedIdentifier;
use glsl::syntax::ArraySpecifier;
use glsl::syntax::TypeName;
use glsl::syntax::StructSpecifier;
use glsl::syntax::UnaryOp;
use glsl::syntax::BinaryOp;
use glsl::syntax::AssignmentOp;
use glsl::syntax::Identifier;
use crate::hir::Type::Function;
use crate::hir::Initializer::Simple;
use crate::hir::SimpleStatement::Jump;

#[derive(Debug)]
pub struct Symbol {
    pub name: String,
    ty: Type
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    ret: Box<TypeSpecifier>,
    params: Vec<TypeSpecifier>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    signatures: NonEmpty<FunctionSignature>
}

#[derive(Clone, Debug, PartialEq)]
pub enum StorageClass {
    None,
    In,
    Out,
    Uniform,
}

/// Fully specified type.
#[derive(Clone, Debug, PartialEq)]
pub struct FullySpecifiedType {
    pub qualifier: Option<TypeQualifier>,
    pub ty: TypeSpecifier
}

impl FullySpecifiedType {
    pub fn new(ty: TypeSpecifierNonArray) -> Self {
        FullySpecifiedType {
            qualifier: None,
            ty: TypeSpecifier {
                ty,
                array_specifier: None
            }
        }
    }


}

impl From<syntax::FullySpecifiedType> for FullySpecifiedType {
    fn from(ty: syntax::FullySpecifiedType) -> Self {
        FullySpecifiedType {
            qualifier: ty.qualifier,
            ty: ty.ty
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Function(FunctionType),
    Variable(StorageClass, FullySpecifiedType),
    Struct(FullySpecifiedType)
}

impl Type {
    fn var(t: TypeSpecifierNonArray) -> Self {
        Type::Variable(StorageClass::None, FullySpecifiedType::new(t))
    }
}

#[derive(Clone, Debug, PartialEq, Copy, Eq, Hash)]
pub struct SymRef(u32);

#[derive(Debug)]
struct Scope {
    name: String,
    names: HashMap<String, SymRef>,
}
impl Scope {
    fn new(name: String) -> Self {  Scope { name, names: HashMap::new() }}
}

#[derive(Debug)]
pub struct State {
    scopes: Vec<Scope>,
    syms: Vec<Symbol>,
}


impl State {
    pub fn new() -> Self {
        State { scopes: Vec::new(), syms: Vec::new() }
    }

    fn lookup(&self, name: &str) -> Option<SymRef> {
        for s in self.scopes.iter().rev() {
            if let Some(sym) = s.names.get(name) {
                return Some(*sym);
            }
        }
        return None;
    }

    fn declare(&mut self, name: &str, ty: Type) -> SymRef {
        let s = SymRef(self.syms.len() as u32);
        self.syms.push(Symbol{ name: name.into(), ty});
        self.scopes.last_mut().unwrap().names.insert(name.into(), s);
        s
    }

    pub fn sym(&self, sym: SymRef) -> &Symbol {
        &self.syms[sym.0 as usize]
    }

    pub fn lookup_sym_mut(&mut self, name: &str) -> Option<&mut Symbol> {
        self.lookup(name).map(move |x| &mut self.syms[x.0 as usize])
    }

    fn push_scope(&mut self, name: String) {
        self.scopes.push(Scope::new(name));
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

/// A declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum Declaration {
    FunctionPrototype(FunctionPrototype),
    InitDeclaratorList(InitDeclaratorList),
    Precision(PrecisionQualifier, TypeSpecifier),
    Block(Block),
    Global(TypeQualifier, Vec<Identifier>)
}

/// A general purpose block, containing fields and possibly a list of declared identifiers. Semantic
/// is given with the storage qualifier.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub qualifier: TypeQualifier,
    pub name: Identifier,
    pub fields: Vec<StructFieldSpecifier>,
    pub identifier: Option<ArrayedIdentifier>
}

/// Function identifier.
#[derive(Clone, Debug, PartialEq)]
pub enum FunIdentifier {
    Identifier(SymRef),
    Expr(Box<Expr>)
}

/// Function prototype.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionPrototype {
    pub ty: FullySpecifiedType,
    pub name: Identifier,
    pub parameters: Vec<FunctionParameterDeclaration>
}

/// Function parameter declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionParameterDeclaration {
    Named(Option<TypeQualifier>, FunctionParameterDeclarator),
    Unnamed(Option<TypeQualifier>, TypeSpecifier)
}

impl FunctionParameterDeclaration {
    /// Create a named function argument.
    pub fn new_named<I, T>(
        ident: I,
        ty: T
    ) -> Self
        where I: Into<ArrayedIdentifier>,
              T: Into<TypeSpecifier> {
        let declator = FunctionParameterDeclarator {
            ty: ty.into(),
            ident: ident.into()
        };

        FunctionParameterDeclaration::Named(None, declator)
    }

    /// Create an unnamed function argument (mostly useful for interfaces / function prototypes).
    pub fn new_unnamed<T>(ty: T) -> Self where T: Into<TypeSpecifier> {
        FunctionParameterDeclaration::Unnamed(None, ty.into())
    }
}

/// Function parameter declarator.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionParameterDeclarator {
    pub ty: TypeSpecifier,
    pub ident: ArrayedIdentifier
}

/// Init declarator list.
#[derive(Clone, Debug, PartialEq)]
pub struct InitDeclaratorList {
    // XXX it feels like separating out the type and the names is better than
    // head and tail
    pub head: SingleDeclaration,
    pub tail: Vec<SingleDeclarationNoType>
}

/// Single declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclaration {
    pub ty: FullySpecifiedType,
    pub name: Option<SymRef>,
    pub array_specifier: Option<ArraySpecifier>,
    pub initializer: Option<Initializer>
}

/// A single declaration with implicit, already-defined type.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleDeclarationNoType {
    pub ident: ArrayedIdentifier,
    pub initializer: Option<Initializer>
}

/// Initializer.
#[derive(Clone, Debug, PartialEq)]
pub enum Initializer {
    Simple(Box<Expr>),
    List(NonEmpty<Initializer>)
}

impl From<Expr> for Initializer {
    fn from(e: Expr) -> Self {
        Initializer::Simple(Box::new(e))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: TypeSpecifier
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldSet {
    Rgba,
    Xyzw,
    Stpq,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwizzleSelector {
    pub field_set: FieldSet,
    pub components: Vec<i8>
}

impl SwizzleSelector {
    fn parse(s: &str) -> Self {
        let mut components = Vec::new();
        let mut field_set = Vec::new();

        for c in s.chars() {
            match c {
                'r' => { components.push(0); field_set.push(FieldSet::Rgba); }
                'x' => { components.push(0); field_set.push(FieldSet::Xyzw); }
                's' => { components.push(0); field_set.push(FieldSet::Stpq); }

                'g' => { components.push(1); field_set.push(FieldSet::Rgba); }
                'y' => { components.push(1); field_set.push(FieldSet::Xyzw); }
                't' => { components.push(1); field_set.push(FieldSet::Stpq); }

                'b' => { components.push(2); field_set.push(FieldSet::Rgba); }
                'z' => { components.push(2); field_set.push(FieldSet::Xyzw); }
                'p' => { components.push(2); field_set.push(FieldSet::Stpq); }

                'a' => { components.push(3); field_set.push(FieldSet::Rgba); }
                'w' => { components.push(3); field_set.push(FieldSet::Xyzw); }
                'q' => { components.push(3); field_set.push(FieldSet::Stpq); }
                _ => panic!("bad selector")
            }
        }

        let first = &field_set[0];
        assert!(field_set.iter().all(|item| item == first));
        assert!(components.len() <= 4);
        SwizzleSelector { field_set: first.clone(), components }
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        let fs = match self.field_set {
            FieldSet::Rgba => ['r','g','b','a'],
            FieldSet::Xyzw => ['x', 'y','z','w'],
            FieldSet::Stpq => ['s','t','p','q'],
        };
        for i in &self.components {
            s.push(fs[*i as usize])
        }
        s
    }
}

/// The most general form of an expression. As you can see if you read the variant list, in GLSL, an
/// assignment is an expression. This is a bit silly but think of an assignment as a statement first
/// then an expression which evaluates to what the statement “returns”.
///
/// An expression is either an assignment or a list (comma) of assignments.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    /// A variable expression, using an identifier.
    Variable(SymRef),
    /// Integral constant expression.
    IntConst(i32),
    /// Unsigned integral constant expression.
    UIntConst(u32),
    /// Boolean constant expression.
    BoolConst(bool),
    /// Single precision floating expression.
    FloatConst(f32),
    /// Double precision floating expression.
    DoubleConst(f64),
    /// A unary expression, gathering a single expression and a unary operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary expression, gathering two expressions and a binary operator.
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// A ternary conditional expression, gathering three expressions.
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    /// An assignment is also an expression. Gathers an expression that defines what to assign to, an
    /// assignment operator and the value to associate with.
    Assignment(Box<Expr>, AssignmentOp, Box<Expr>),
    /// Add an array specifier to an expression.
    Bracket(Box<Expr>, ArraySpecifier),
    /// A functional call. It has a function identifier and a list of expressions (arguments).
    FunCall(FunIdentifier, Vec<Expr>),
    /// An expression associated with a field selection (struct).
    Dot(Box<Expr>, Identifier),
    /// An expression associated with a component selection
    SwizzleSelector(Box<Expr>, SwizzleSelector),
    /// Post-incrementation of an expression.
    PostInc(Box<Expr>),
    /// Post-decrementation of an expression.
    PostDec(Box<Expr>),
    /// An expression that contains several, separated with comma.
    Comma(Box<Expr>, Box<Expr>)
}

/*
impl From<i32> for Expr {
    fn from(x: i32) -> Expr {
        ExprKind::IntConst(x)
    }
}

impl From<u32> for Expr {
    fn from(x: u32) -> Expr {
        Expr::UIntConst(x)
    }
}

impl From<bool> for Expr {
    fn from(x: bool) -> Expr {
        Expr::BoolConst(x)
    }
}

impl From<f32> for Expr {
    fn from(x: f32) -> Expr {
        Expr::FloatConst(x)
    }
}

impl From<f64> for Expr {
    fn from(x: f64) -> Expr {
        Expr::DoubleConst(x)
    }
}
*/
/// Starting rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TranslationUnit(pub NonEmpty<ExternalDeclaration>);

impl TranslationUnit {
    /// Construct a translation unit from an iterator.
    ///
    /// # Errors
    ///
    /// `None` if the iterator yields no value.
    pub fn from_iter<I>(iter: I) -> Option<Self> where I: IntoIterator<Item = ExternalDeclaration> {
        NonEmpty::from_iter(iter).map(TranslationUnit)
    }
}

impl Deref for TranslationUnit {
    type Target = NonEmpty<ExternalDeclaration>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TranslationUnit {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for TranslationUnit {
    type IntoIter = <NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TranslationUnit {
    type IntoIter = <&'a NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

impl<'a> IntoIterator for &'a mut TranslationUnit {
    type IntoIter = <&'a mut NonEmpty<ExternalDeclaration> as IntoIterator>::IntoIter;
    type Item = &'a mut ExternalDeclaration;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.0).into_iter()
    }
}

/// External declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalDeclaration {
    Preprocessor(Preprocessor),
    FunctionDefinition(FunctionDefinition),
    Declaration(Declaration)
}

impl ExternalDeclaration {
    /// Create a new function.
    pub fn new_fn<T, N, A, S>(
        ret_ty: T,
        name: N,
        args: A,
        body: S
    ) -> Self
        where T: Into<FullySpecifiedType>,
              N: Into<Identifier>,
              A: IntoIterator<Item = FunctionParameterDeclaration>,
              S: IntoIterator<Item = Statement> {
        ExternalDeclaration::FunctionDefinition(
            FunctionDefinition {
                prototype: FunctionPrototype {
                    ty: ret_ty.into(),
                    name: name.into(),
                    parameters: args.into_iter().collect()
                },
                statement: CompoundStatement {
                    statement_list: body.into_iter().collect()
                }
            }
        )
    }

    /// Create a new structure.
    ///
    /// # Errors
    ///
    ///   - [`None`] if no fields are provided. GLSL forbids having empty structs.
    pub fn new_struct<N, F>(name: N, fields: F) -> Option<Self>
        where N: Into<TypeName>,
              F: IntoIterator<Item = StructFieldSpecifier> {
        let fields: Vec<_> = fields.into_iter().collect();

        if fields.is_empty() {
            None
        } else {
            Some(ExternalDeclaration::Declaration(
                Declaration::InitDeclaratorList(
                    InitDeclaratorList {
                        head: SingleDeclaration {
                            ty: FullySpecifiedType {
                                qualifier: None,
                                ty: TypeSpecifier {
                                    ty: TypeSpecifierNonArray::Struct(
                                        StructSpecifier {
                                            name: Some(name.into()),
                                            fields: NonEmpty(fields.into_iter().collect())
                                        }
                                    ),
                                    array_specifier: None
                                }
                            },
                            name: None,
                            array_specifier: None,
                            initializer: None
                        },
                        tail: vec![]
                    }
                )
            ))
        }
    }
}

/// Function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionDefinition {
    pub prototype: FunctionPrototype,
    pub statement: CompoundStatement,
}

/// Compound statement (with no new scope).
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundStatement {
    pub statement_list: Vec<Statement>
}

impl FromIterator<Statement> for CompoundStatement {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = Statement> {
        CompoundStatement {
            statement_list: iter.into_iter().collect()
        }
    }
}

/// Statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Compound(Box<CompoundStatement>),
    Simple(Box<SimpleStatement>)
}

/// Simple statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SimpleStatement {
    Declaration(Declaration),
    Expression(ExprStatement),
    Selection(SelectionStatement),
    Switch(SwitchStatement),
    Iteration(IterationStatement),
    Jump(JumpStatement)
}

impl SimpleStatement {
    /// Create a new expression statement.
    pub fn new_expr<E>(expr: E) -> Self where E: Into<Expr> {
        SimpleStatement::Expression(Some(expr.into()))
    }

    /// Create a new selection statement (if / else).
    pub fn new_if_else<If, True, False>(
        ife: If,
        truee: True,
        falsee: False
    ) -> Self
        where If: Into<Expr>,
              True: Into<Statement>,
              False: Into<Statement> {
        SimpleStatement::Selection(
            SelectionStatement {
                cond: Box::new(ife.into()),
                rest: SelectionRestStatement::Else(Box::new(truee.into()), Box::new(falsee.into()))
            }
        )
    }

    /// Create a new while statement.
    pub fn new_while<C, S>(
        cond: C,
        body: S
    ) -> Self
        where C: Into<Condition>,
              S: Into<Statement> {
        SimpleStatement::Iteration(
            IterationStatement::While(cond.into(), Box::new(body.into()))
        )
    }

    /// Create a new do-while statement.
    pub fn new_do_while<C, S>(
        body: S,
        cond: C
    ) -> Self
        where S: Into<Statement>,
              C: Into<Expr> {
        SimpleStatement::Iteration(
            IterationStatement::DoWhile(Box::new(body.into()), Box::new(cond.into()))
        )
    }
}

/// Expression statement.
pub type ExprStatement = Option<Expr>;

/// Selection statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionStatement {
    pub cond: Box<Expr>,
    pub rest: SelectionRestStatement
}

/// Condition.
#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    Expr(Box<Expr>),
    Assignment(FullySpecifiedType, Identifier, Initializer)
}

impl From<Expr> for Condition {
    fn from(expr: Expr) -> Self {
        Condition::Expr(Box::new(expr))
    }
}

/// Selection rest statement.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectionRestStatement {
    /// Body of the if.
    Statement(Box<Statement>),
    /// The first argument is the body of the if, the rest is the next statement.
    Else(Box<Statement>, Box<Statement>)
}

/// Switch statement.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchStatement {
    pub head: Box<Expr>,
    pub cases: Vec<Case>
}

/// Case label statement.
#[derive(Clone, Debug, PartialEq)]
pub enum CaseLabel {
    Case(Box<Expr>),
    Def
}

/// An individual case
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    pub label: CaseLabel,
    pub stmts: Vec<Statement>
}

/// Iteration statement.
#[derive(Clone, Debug, PartialEq)]
pub enum IterationStatement {
    While(Condition, Box<Statement>),
    DoWhile(Box<Statement>, Box<Expr>),
    For(ForInitStatement, ForRestStatement, Box<Statement>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub enum ForInitStatement {
    Expression(Option<Expr>),
    Declaration(Box<Declaration>)
}

/// For init statement.
#[derive(Clone, Debug, PartialEq)]
pub struct ForRestStatement {
    pub condition: Option<Condition>,
    pub post_expr: Option<Box<Expr>>
}

/// Jump statement.
#[derive(Clone, Debug, PartialEq)]
pub enum JumpStatement {
    Continue,
    Break,
    Return(Box<Expr>),
    Discard
}

/// Some basic preprocessor commands.
///
/// As it’s important to carry them around the AST because they cannot be substituted in a normal
/// preprocessor (they’re used by GPU’s compilers), those preprocessor commands are available for
/// inspection.
///
/// > Important note: so far, only `#version` and `#extension` are supported. Other pragmas will be
/// > added in the future. Stay tuned.
#[derive(Clone, Debug, PartialEq)]
pub enum Preprocessor {
    Define(PreprocessorDefine),
    Version(PreprocessorVersion),
    Extension(PreprocessorExtension)
}

/// A #define preprocessor command.
/// Allows any expression but only Integer and Float literals make sense
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorDefine {
    pub name: Identifier,
    pub value: Expr,
}

/// A #version preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorVersion {
    pub version: u16,
    pub profile: Option<PreprocessorVersionProfile>
}

/// A #version profile annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorVersionProfile {
    Core,
    Compatibility,
    ES
}

/// An #extension preprocessor command.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessorExtension {
    pub name: PreprocessorExtensionName,
    pub behavior: Option<PreprocessorExtensionBehavior>
}

/// An #extension name annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionName {
    /// All extensions you could ever imagine in your whole lifetime (how crazy is that!).
    All,
    /// A specific extension.
    Specific(String)
}

/// An #extension behavior annotation.
#[derive(Clone, Debug, PartialEq)]
pub enum PreprocessorExtensionBehavior {
    Require,
    Enable,
    Warn,
    Disable
}

trait NonEmptyExt<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, f: F) -> NonEmpty<U>;
    fn new(x: T) -> NonEmpty<T>;
}

impl<T> NonEmptyExt<T> for NonEmpty<T> {
    fn map<U, F: FnMut(&mut State, &T) -> U>(&self, s: &mut State, mut f: F) -> NonEmpty<U> {
        NonEmpty::from_iter(self.into_iter().map(|x| f(s, &x))).unwrap()
    }
    fn new(x: T) -> NonEmpty<T> {
        NonEmpty::from_iter(vec![x].into_iter()).unwrap()
    }
}


fn translate_initializater(state: &mut State, i: &syntax::Initializer) -> Initializer {
    match i {
        syntax::Initializer::Simple(i) => Initializer::Simple(Box::new(translate_expression(state, i))),
        _ => panic!()
    }
}

fn translate_single_declaration(state: &mut State, d: &syntax::SingleDeclaration) -> SingleDeclaration {
    let mut ty = d.ty.clone();
    ty.ty.array_specifier = d.array_specifier.clone();
    let sym = match &ty.ty.ty {
        TypeSpecifierNonArray::Struct(s) => {
            state.declare(s.name.as_ref().unwrap().as_str(), Type::Struct(ty.clone().into()))
        }
        _ => {
            let mut storage = StorageClass::None;
            for qual in ty.qualifier.iter().flat_map(|x| x.qualifiers.0.iter()) {
                match qual {
                    syntax::TypeQualifierSpec::Storage(s) => {
                        match (&storage, s) {
                            (StorageClass::None, syntax::StorageQualifier::Out) => {
                                storage = StorageClass::Out
                            }
                            (StorageClass::None, syntax::StorageQualifier::In) => {
                                storage = StorageClass::In
                            }
                            (StorageClass::None, syntax::StorageQualifier::Uniform) => {
                                storage = StorageClass::Uniform
                            }
                            _ => panic!("bad storage {:?}", (storage, s))
                        }
                    }
                    _ => {}
                }
            }
            state.declare(d.name.as_ref().unwrap().as_str(), Type::Variable(storage, ty.clone().into()))
        }
    };
    SingleDeclaration {
        name: d.name.as_ref().and(Some(sym)),
        ty: ty.into(),
        array_specifier: d.array_specifier.clone(),
        initializer: d.initializer.as_ref().map(|x| translate_initializater(state, x)),
    }
}

fn translate_single_declaration_no_type(state: &mut State, d: &syntax::SingleDeclarationNoType) -> SingleDeclarationNoType {
    panic!()
}

fn translate_init_declarator_list(state: &mut State, l: &syntax::InitDeclaratorList) -> InitDeclaratorList {
    InitDeclaratorList {
        head: translate_single_declaration(state, &l.head),
        tail: l.tail.iter().map(|x| translate_single_declaration_no_type(state, x)).collect()
    }
}

fn translate_declaration(state: &mut State, d: &syntax::Declaration) -> Declaration {
    match d {
        syntax::Declaration::Block(b) => Declaration::Block(panic!()),
        syntax::Declaration::FunctionPrototype(p) => Declaration::FunctionPrototype(panic!()),
        syntax::Declaration::Global(ty, ids) => Declaration::Global(panic!(), panic!()),
        syntax::Declaration::InitDeclaratorList(dl) => Declaration::InitDeclaratorList(translate_init_declarator_list(state, dl)),
        syntax::Declaration::Precision(p, ts) => Declaration::Precision(panic!(), panic!()),
    }
}

fn is_vector(ty: &TypeSpecifier) -> bool {
    match ty.ty {
        TypeSpecifierNonArray::Vec3 | TypeSpecifierNonArray::Vec2 | TypeSpecifierNonArray::Vec4 => {
            true
        }
        _ => false
    }
}

fn compatible_type(lhs: &TypeSpecifier, rhs: &TypeSpecifier) -> bool {
    if lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double) &&
        rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) {
        true
    } else if rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double) &&
        lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) {
        true
    } else {
        lhs == rhs
    }
}

fn promoted_type(lhs: &TypeSpecifier, rhs: &TypeSpecifier) -> TypeSpecifier {
    if lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double) &&
        rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) {
        TypeSpecifier::new(TypeSpecifierNonArray::Double)
    } else if lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) &&
        rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double) {
        TypeSpecifier::new(TypeSpecifierNonArray::Double)
    } else if is_vector(&lhs) && (rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) ||
        rhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double)) {
        // scalars promote to vectors
        lhs.clone()
    } else if is_vector(&rhs) && (lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Float) ||
        lhs == &TypeSpecifier::new(TypeSpecifierNonArray::Double)) {
        // scalars promote to vectors
        rhs.clone()
    } else {
        assert_eq!(lhs, rhs);
        lhs.clone()
    }
}

fn translate_expression(state: &mut State, e: &syntax::Expr) -> Expr {
    match e {
        syntax::Expr::Variable(i) => {
            let sym = match state.lookup(i.as_str()) {
                Some(sym) => sym,
                None => panic!("missing declaration {}", i.as_str())
            };
            let ty = match &state.sym(sym).ty {
                Type::Variable(_, ty) => ty.ty.clone(),
                _ => panic!("bad variable type")
            };
            Expr { kind: ExprKind::Variable(sym), ty }
        },
        syntax::Expr::Assignment(lhs, op, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert!(compatible_type(&lhs.ty, &rhs.ty));
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Assignment(lhs, op.clone(), rhs), ty }
        }
        syntax::Expr::Binary(op, lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            let ty = if op == &BinaryOp::Mult {
                if lhs.ty.ty == TypeSpecifierNonArray::Mat3 && rhs.ty.ty == TypeSpecifierNonArray::Vec3 {
                    rhs.ty.clone()
                } else {
                    promoted_type(&lhs.ty, &rhs.ty)
                }
            } else {
                promoted_type(&lhs.ty, &rhs.ty)
            };
            Expr { kind: ExprKind::Binary(op.clone(), lhs, rhs), ty}
        }
        syntax::Expr::Unary(op, e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::Unary(op.clone(), e), ty}
        }
        syntax::Expr::BoolConst(b) => {
            Expr { kind: ExprKind::BoolConst(*b), ty: TypeSpecifier::new(TypeSpecifierNonArray::Bool) }
        }
        syntax::Expr::Comma(lhs, rhs) => {
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Comma(lhs, rhs), ty }
        }
        syntax::Expr::DoubleConst(d) => {
            Expr { kind: ExprKind::DoubleConst(*d), ty: TypeSpecifier::new(TypeSpecifierNonArray::Double) }
        }
        syntax::Expr::FloatConst(f) => {
            Expr { kind: ExprKind::FloatConst(*f), ty: TypeSpecifier::new(TypeSpecifierNonArray::Float) }
        },
        syntax::Expr::FunCall(fun, params) => {
            let ret_ty: TypeSpecifier;
            let params: Vec<Expr> = params.iter().map(|x| translate_expression(state, x)).collect();
            Expr {
                kind:
                ExprKind::FunCall(
                    match fun {
                        syntax::FunIdentifier::Identifier(i) => {
                            let sym = match state.lookup(i.as_str()) {
                                Some(s) => s,
                                None => panic!("missing {}", i.as_str())
                            };
                            match &state.sym(sym).ty {
                                Type::Function(fn_ty) => {
                                    let mut ret = None;
                                    for sig in &fn_ty.signatures {
                                        let mut matching = true;
                                        for (e, p) in params.iter().zip(sig.params.iter()) {
                                            if !compatible_type(&e.ty, p) {
                                                matching = false;
                                                break;
                                            }
                                        }
                                        if matching {
                                            ret = Some(sig.ret.clone());
                                            break;
                                        }
                                    }
                                    ret_ty = match ret {
                                        Some(t) => *t,
                                        None => {
                                            dbg!(&fn_ty.signatures);
                                            dbg!(params.iter().map(|p| &p.ty).collect::<Vec<_>>());
                                            panic!("no matching func {}", i.as_str())
                                        }
                                    };
                                },
                                Type::Struct(t) => {
                                    ret_ty = t.ty.clone()
                                }
                                _ => panic!("can only call functions")
                            };

                            FunIdentifier::Identifier(sym)
                        },
                        _ => panic!()
                    },
                    params
                ),
                ty: ret_ty,
            }
        }
        syntax::Expr::IntConst(i) => {
            Expr { kind: ExprKind::IntConst(*i), ty: TypeSpecifier::new(TypeSpecifierNonArray::Int) }
        }
        syntax::Expr::UIntConst(u) => {
            Expr { kind: ExprKind::UIntConst(*u), ty: TypeSpecifier::new(TypeSpecifierNonArray::UInt) }
        }
        syntax::Expr::PostDec(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostDec(e), ty }
        }
        syntax::Expr::PostInc(e) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            Expr { kind: ExprKind::PostInc(e), ty }
        }
        syntax::Expr::Ternary(cond, lhs, rhs) => {
            let cond = Box::new(translate_expression(state, cond));
            let lhs = Box::new(translate_expression(state, lhs));
            let rhs = Box::new(translate_expression(state, rhs));
            assert_eq!(lhs.ty, rhs.ty);
            let ty = lhs.ty.clone();
            Expr { kind: ExprKind::Ternary(cond, lhs, rhs), ty }
        }
        syntax::Expr::Dot(e, i) => {
            let e = Box::new(translate_expression(state, e));
            let ty = e.ty.clone();
            if is_vector(&ty) {
                let ty = TypeSpecifier::new(match i.as_str().len() {
                    1 => TypeSpecifierNonArray::Float,
                    2 => TypeSpecifierNonArray::Vec2,
                    3 => TypeSpecifierNonArray::Vec3,
                    4 => TypeSpecifierNonArray::Vec4,
                    _ => panic!(),
                });

                Expr { kind: ExprKind::SwizzleSelector(e, SwizzleSelector::parse(i.as_str())), ty }
            } else {
                panic!();
                Expr { kind: ExprKind::Dot(e, i.clone()), ty }
            }
        }
        syntax::Expr::Bracket(e, specifier) =>{
            let e = Box::new(translate_expression(state, e));
            let ty = if is_vector(&e.ty) {
                TypeSpecifier::new(TypeSpecifierNonArray::Float)
            } else {
                assert!(e.ty.array_specifier.is_some());
                e.ty.clone()
            };
            Expr { kind: ExprKind::Bracket(e, specifier.clone()), ty }
        }
    }
}

fn translate_switch(state: &mut State, s: &syntax::SwitchStatement) -> SwitchStatement {
    let mut cases = Vec::new();

    let mut case = None;
    for stmt in &s.body {
        match stmt {
            syntax::Statement::Simple(s) => {
                match &**s {
                    syntax::SimpleStatement::CaseLabel(label) => {
                        match case.take() {
                            Some(case) => cases.push(case),
                            _ => {}
                        }
                        case = Some(Case { label: translate_case(state, &label), stmts: Vec::new() })
                    }
                    _ => {
                        match case {
                            Some(ref mut case) => case.stmts.push(translate_statement(state, stmt)),
                            _ => panic!("switch must start with case")
                        }
                    }
                }
            }
            _ => {
                match case {
                    Some(ref mut case) => case.stmts.push(translate_statement(state, stmt)),
                    _ => panic!("switch must start with case")
                }
            }
        }

    }
    match case.take() {
        Some(case) => cases.push(case),
        _ => {}
    }
    SwitchStatement {
        head: Box::new(translate_expression(state, &s.head)),
        cases
    }
}

fn translate_jump(state: &mut State, s: &syntax::JumpStatement) -> JumpStatement {
    match s {
        syntax::JumpStatement::Break => JumpStatement::Break,
        syntax::JumpStatement::Continue => JumpStatement::Continue,
        syntax::JumpStatement::Discard => JumpStatement::Discard,
        syntax::JumpStatement::Return(e) => JumpStatement::Return(Box::new(translate_expression(state, e)))
    }
}

fn translate_condition(state: &mut State, c: &syntax::Condition) -> Condition {
    match c {
        syntax::Condition::Expr(e) => Condition::Expr(Box::new(translate_expression(state, e))),
        _ => panic!()
    }
}

fn translate_for_init(state: &mut State, s: &syntax::ForInitStatement) -> ForInitStatement {
    match s {
        syntax::ForInitStatement::Expression(e) => ForInitStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::ForInitStatement::Declaration(d) => ForInitStatement::Declaration(Box::new(translate_declaration(state, d))),
    }
}

fn translate_for_rest(state: &mut State, s: &syntax::ForRestStatement) -> ForRestStatement {
    ForRestStatement {
        condition: s.condition.as_ref().map(|c| translate_condition(state, c)),
        post_expr: s.post_expr.as_ref().map(|e| Box::new(translate_expression(state, e)))
    }
}

fn translate_iteration(state: &mut State, s: &syntax::IterationStatement) -> IterationStatement {
    match s {
        syntax::IterationStatement::While(cond, s) =>
            IterationStatement::While(translate_condition(state, cond), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::For(init, rest, s) =>
            IterationStatement::For(translate_for_init(state, init),translate_for_rest(state, rest), Box::new(translate_statement(state, s))),
        syntax::IterationStatement::DoWhile(s, e) =>
            IterationStatement::DoWhile(Box::new(translate_statement(state, s)), Box::new(translate_expression(state, e))),
    }
}

fn translate_case(state: &mut State, c: &syntax::CaseLabel) -> CaseLabel {
    match c {
        syntax::CaseLabel::Def => CaseLabel::Def,
        syntax::CaseLabel::Case(e) => CaseLabel::Case(Box::new(translate_expression(state, e)))
    }
}

fn translate_selection_rest(state: &mut State, s: &syntax::SelectionRestStatement) -> SelectionRestStatement {
    match s {
        syntax::SelectionRestStatement::Statement(s) => SelectionRestStatement::Statement(Box::new(translate_statement(state, s))),
        syntax::SelectionRestStatement::Else(if_body, rest) => {
            SelectionRestStatement::Else(Box::new(translate_statement(state, if_body)), Box::new(translate_statement(state, rest)))
        }
    }
}

fn translate_selection(state: &mut State, s: &syntax::SelectionStatement) -> SelectionStatement {
    SelectionStatement {
        cond: Box::new(translate_expression(state, &s.cond)),
        rest: translate_selection_rest(state, &s.rest),
    }
}

fn translate_simple_statement(state: &mut State, s: &syntax::SimpleStatement) -> SimpleStatement {
    match s {
        syntax::SimpleStatement::Declaration(d) => SimpleStatement::Declaration(translate_declaration(state, d)),
        syntax::SimpleStatement::Expression(e) => SimpleStatement::Expression(e.as_ref().map(|e| translate_expression(state, e))),
        syntax::SimpleStatement::Iteration(i) => SimpleStatement::Iteration(translate_iteration(state, i)),
        syntax::SimpleStatement::Selection(s) => SimpleStatement::Selection(translate_selection(state, s)),
        syntax::SimpleStatement::Jump(j) => SimpleStatement::Jump(translate_jump(state, j)),
        syntax::SimpleStatement::Switch(s) => SimpleStatement::Switch(translate_switch(state, s)),
        syntax::SimpleStatement::CaseLabel(s) => panic!("should be handled by translate_switch")
    }
}

fn translate_statement(state: &mut State, s: &syntax::Statement) -> Statement {
    match s {
        syntax::Statement::Compound(s) => Statement::Compound(Box::new(translate_compound_statement(state, s))),
        syntax::Statement::Simple(s) => Statement::Simple(Box::new(translate_simple_statement(state, s)))
    }
}


fn translate_compound_statement(state: &mut State, cs: &syntax::CompoundStatement) -> CompoundStatement {
    CompoundStatement { statement_list: cs.statement_list.iter().map(|x| translate_statement(state, x)).collect() }
}

fn translate_function_parameter_declarator(state: &mut State, d: &syntax::FunctionParameterDeclarator) -> FunctionParameterDeclarator {
    FunctionParameterDeclarator {
        ty: d.ty.clone(),
        ident: d.ident.clone(),
    }
}

fn translate_function_parameter_declaration(state: &mut State, p: &syntax::FunctionParameterDeclaration) ->
  FunctionParameterDeclaration
{
    match p {
        syntax::FunctionParameterDeclaration::Named(qual, p) => {
            state.declare(p.ident.ident.as_str(), Type::Variable(
                StorageClass::None,
                FullySpecifiedType {
                    qualifier: None,
                    ty: TypeSpecifier {
                        ty: p.ty.ty.clone(),
                        array_specifier: None
                    }
                }));
            FunctionParameterDeclaration::Named(qual.clone(), translate_function_parameter_declarator(state, p))
        }
        syntax::FunctionParameterDeclaration::Unnamed(qual, p) => {
            FunctionParameterDeclaration::Unnamed(qual.clone(), p.clone())
        }

    }
}

fn translate_prototype(state: &mut State, cs: &syntax::FunctionPrototype) -> FunctionPrototype {
    FunctionPrototype {
        ty: cs.ty.clone().into(),
        name: cs.name.clone(),
        parameters: cs.parameters.iter().map(|x| translate_function_parameter_declaration(state, x)).collect(),
    }
}

fn translate_function_definition(state: &mut State, fd: &syntax::FunctionDefinition) -> FunctionDefinition {
    let prototype = translate_prototype(state, &fd.prototype);
    let params = prototype.parameters.iter().map(|p| match p {
        FunctionParameterDeclaration::Named(_, p) => p.ty.clone(),
        FunctionParameterDeclaration::Unnamed(_, p) => p.clone(),
    }).collect();
    let sig = FunctionSignature{ ret: Box::new(prototype.ty.ty.clone()), params };
    state.declare(fd.prototype.name.as_str(), Type::Function(FunctionType{ signatures: NonEmpty::new(sig)}));
    state.push_scope(fd.prototype.name.as_str().into());
    let f = FunctionDefinition {
        prototype,
        statement: translate_compound_statement(state, &fd.statement)
    };
    state.pop_scope();
    f
}

fn translate_external_declaration(state: &mut State, ed: &syntax::ExternalDeclaration) -> ExternalDeclaration {
    match ed {
        syntax::ExternalDeclaration::Declaration(d) =>
            ExternalDeclaration::Declaration(translate_declaration(state, d)),
        syntax::ExternalDeclaration::FunctionDefinition(fd) =>
            ExternalDeclaration::FunctionDefinition(translate_function_definition(state, fd)),
        syntax::ExternalDeclaration::Preprocessor(p) =>
            ExternalDeclaration::Preprocessor(panic!())
    }
}

fn declare_function(state: &mut State, name: &str, ret: TypeSpecifier, params: Vec<TypeSpecifier>) {
    let sig = FunctionSignature{ ret: Box::new(ret), params };
    match state.lookup_sym_mut(name) {
        Some(Symbol { ty: Type::Function(f), ..}) => f.signatures.push(sig),
        None => { state.declare(name, Type::Function(FunctionType{ signatures: NonEmpty::new(sig)})); },
        _ => panic!("overloaded function name {}", name)
    }
    //state.declare(name, Type::Function(FunctionType{ v}))
}


pub fn ast_to_hir(state: &mut State, tu: &syntax::TranslationUnit) -> TranslationUnit {
    // global scope
    state.push_scope("global".into());
    use TypeSpecifierNonArray::*;
    declare_function(state, "vec3", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Float), TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "vec3", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Float)]);
    declare_function(state, "vec3", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Vec2), TypeSpecifier::new(Float)]);
    declare_function(state, "vec4", TypeSpecifier::new(Vec4),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Float)]);
    declare_function(state, "vec4", TypeSpecifier::new(Vec4),
                     vec![TypeSpecifier::new(Float), TypeSpecifier::new(Float), TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "vec2", TypeSpecifier::new(Vec2),
                     vec![TypeSpecifier::new(Float)]);
    declare_function(state, "mix", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3)]);
    declare_function(state, "mix", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3), TypeSpecifier::new(Float)]);
    declare_function(state, "mix", TypeSpecifier::new(Float),
                     vec![TypeSpecifier::new(Float), TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "step", TypeSpecifier::new(Vec2),
                     vec![TypeSpecifier::new(Vec2), TypeSpecifier::new(Vec2)]);
    declare_function(state, "max", TypeSpecifier::new(Vec2),
                     vec![TypeSpecifier::new(Vec2), TypeSpecifier::new(Vec2)]);
    declare_function(state, "max", TypeSpecifier::new(Float),
                     vec![TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "min", TypeSpecifier::new(Float),
                     vec![TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "fwidth", TypeSpecifier::new(Vec2),
                     vec![TypeSpecifier::new(Vec2)]);
    declare_function(state, "clamp", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Float), TypeSpecifier::new(Float)]);
    declare_function(state, "clamp", TypeSpecifier::new(Double),
                     vec![TypeSpecifier::new(Double), TypeSpecifier::new(Double), TypeSpecifier::new(Double)]);
    declare_function(state, "clamp", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3)]);
    declare_function(state, "length", TypeSpecifier::new(Float), vec![TypeSpecifier::new(Vec2)]);
    declare_function(state, "pow", TypeSpecifier::new(Vec3), vec![TypeSpecifier::new(Vec3)]);
    declare_function(state, "pow", TypeSpecifier::new(Float), vec![TypeSpecifier::new(Float)]);
    declare_function(state, "lessThanEqual", TypeSpecifier::new(BVec3),
                     vec![TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3)]);
    declare_function(state, "if_then_else", TypeSpecifier::new(Vec3),
                     vec![TypeSpecifier::new(BVec3), TypeSpecifier::new(Vec3), TypeSpecifier::new(Vec3)]);
    declare_function(state, "floor", TypeSpecifier::new(Vec4),
                     vec![TypeSpecifier::new(Vec4)]);
    declare_function(state, "floor", TypeSpecifier::new(Double),
                     vec![TypeSpecifier::new(Double)]);
    declare_function(state, "int", TypeSpecifier::new(Int),
                     vec![TypeSpecifier::new(Float)]);
    declare_function(state, "uint", TypeSpecifier::new(UInt),
                     vec![TypeSpecifier::new(Float)]);
    declare_function(state, "uint", TypeSpecifier::new(UInt),
                     vec![TypeSpecifier::new(Int)]);
    declare_function(state, "ivec2", TypeSpecifier::new(IVec2),
                     vec![TypeSpecifier::new(UInt), TypeSpecifier::new(UInt)]);
    declare_function(state, "ivec2", TypeSpecifier::new(IVec2),
                     vec![TypeSpecifier::new(UInt), TypeSpecifier::new(UInt)]);
    declare_function(state, "texelFetch", TypeSpecifier::new(Vec4),
                     vec![TypeSpecifier::new(Sampler2D), TypeSpecifier::new(IVec2), TypeSpecifier::new(Int)]);
    declare_function(state, "texture", TypeSpecifier::new(Vec4),
                     vec![TypeSpecifier::new(Sampler2D), TypeSpecifier::new(Vec3)]);
    state.declare("gl_FragCoord", Type::var(Vec4));
    state.declare("gl_FragColor", Type::var(Vec4));

    TranslationUnit(tu.0.map(state, translate_external_declaration))
}
