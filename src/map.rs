// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;
use std::cmp::Ordering::{self, Less, Equal, Greater};
use std::fmt::{self, Debug};
use std::iter::{self, IntoIterator};
use std::mem::{replace, swap};
use std::ops;
use std::ops::Deref;
use std::ptr;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use compare::{Compare, Natural, natural};

use super::Bound;

/// This is implemented as an AA tree, which is a simplified variation of
/// a red-black tree where red (horizontal) nodes can only be added
/// as a right child. The time complexity is the same, and re-balancing
/// operations are more frequent but also cheaper.
///
/// # Examples
///
/// ```rust
/// use stable_bst::TreeMap;
///
/// let mut map = TreeMap::new();
///
/// map.insert(2, "bar");
/// map.insert(1, "foo");
/// map.insert(3, "quux");
///
/// // In ascending order by keys
/// for (key, value) in map.iter() {
///     println!("{}: {}", key, value);
/// }
///
/// // Prints 1, 2, 3
/// for key in map.keys() {
///     println!("{}", key);
/// }
///
/// // Prints `foo`, `bar`, `quux`
/// for value in map.values() {
///     println!("{}", value);
/// }
///
/// map.remove(&1);
/// assert_eq!(map.len(), 2);
///
/// if !map.contains_key(&1) {
///     println!("1 is no more");
/// }
///
/// for key in 0..4 {
///     match map.get(&key) {
///         Some(value) => println!("{} has a value: {}", key, value),
///         None => println!("{} not in map", key),
///     }
/// }
///
/// map.clear();
/// assert!(map.is_empty());
/// ```
///
/// A `TreeMap` can also be used with a custom ordering:
///
/// ```rust

/// use stable_bst::TreeMap;
///
/// struct Troll<'a> {
///     name: &'a str,
///     level: u32,
/// }
///
/// // Use a map to store trolls, sorted by level, and track a list of
/// // heroes slain.
/// let mut trolls = TreeMap::with_comparator(|l: &Troll, r: &Troll| l.level.cmp(&r.level));
///
/// trolls.insert(Troll { name: "Orgarr", level: 2 },
///               vec!["King Karl"]);
/// trolls.insert(Troll { name: "Blargarr", level: 3 },
///               vec!["Odd"]);
/// trolls.insert(Troll { name: "Kron the Smelly One", level: 4 },
///               vec!["Omar the Brave", "Peter: Slayer of Trolls"]);
/// trolls.insert(Troll { name: "Wartilda", level: 1 },
///               vec![]);
///
/// println!("You are facing {} trolls!", trolls.len());
///
/// // Print the trolls, ordered by level with smallest level first
/// for (troll, heroes) in trolls.iter() {
///     let what = if heroes.len() == 1 { "hero" }
///                else { "heroes" };
///
///     println!("level {}: '{}' has slain {} {}",
///              troll.level, troll.name, heroes.len(), what);
/// }
///
/// // Kill all trolls
/// trolls.clear();
/// assert_eq!(trolls.len(), 0);
/// ```
// Future improvements:
//
// range search - O(log n) retrieval of an iterator from some key
//
// (possibly) implement the overloads Python does for sets:
//   * intersection: &
//   * difference: -
//   * symmetric difference: ^
//   * union: |
// These would be convenient since the methods work like `each`
#[derive(Clone)]
pub struct TreeMap<K, V, C: Compare<K> = Natural<K>> {
    root: Option<Box<TreeNode<K, V>>>,
    length: usize,
    cmp: C,
}

// FIXME: determine what `PartialEq` means for comparator-based `TreeMap`s
impl<K: PartialEq + Ord, V: PartialEq> PartialEq for TreeMap<K, V> {
    #[inline]
    fn eq(&self, other: &TreeMap<K, V>) -> bool {
        self.iter().eq(other)
    }
}

// FIXME: determine what `Eq` means for comparator-based `TreeMap`s
impl<K: Eq + Ord, V: Eq> Eq for TreeMap<K, V> {}

// FIXME: determine what `PartialOrd` means for comparator-based `TreeMap`s
impl<K: Ord, V: PartialOrd> PartialOrd for TreeMap<K, V> {
    #[inline]
    fn partial_cmp(&self, other: &TreeMap<K, V>) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

// FIXME: determine what `Ord` means for comparator-based `TreeMap`s
impl<K: Ord, V: Ord> Ord for TreeMap<K, V> {
    #[inline]
    fn cmp(&self, other: &TreeMap<K, V>) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<K: Debug, V: Debug, C> Debug for TreeMap<K, V, C>
    where C: Compare<K>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "{{"));

        for (i, (k, v)) in self.iter().enumerate() {
            if i != 0 {
                try!(write!(f, ", "));
            }
            try!(write!(f, "{:?}: {:?}", *k, *v));
        }

        write!(f, "}}")
    }
}

impl<K, V, C> Default for TreeMap<K, V, C>
    where C: Compare<K> + Default
{
    #[inline]
    fn default() -> TreeMap<K, V, C> {
        TreeMap::with_comparator(Default::default())
    }
}

impl<'a, K, V, C, Q: ?Sized> ops::Index<&'a Q> for TreeMap<K, V, C>
    where C: Compare<K> + Compare<Q, K>
{
    type Output = V;
    #[inline]
    fn index(&self, i: &'a Q) -> &V {
        self.get(i).expect("no entry found for key")
    }
}

impl<'a, K, V, C, Q: ?Sized> ops::IndexMut<&'a Q> for TreeMap<K, V, C>
    where C: Compare<K> + Compare<Q, K>
{
    #[inline]
    fn index_mut(&mut self, i: &'a Q) -> &mut V {
        self.get_mut(i).expect("no entry found for key")
    }
}

impl<K: Ord, V> TreeMap<K, V> {
    /// Creates an empty `TreeMap` ordered according to the natural order of its keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map: TreeMap<&str, i32> = TreeMap::new();
    /// ```
    pub fn new() -> TreeMap<K, V> {
        TreeMap::with_comparator(natural())
    }
}

impl<K, V, C> TreeMap<K, V, C>
    where C: Compare<K>
{
    /// Creates an empty `TreeMap` ordered according to the given comparator.
    pub fn with_comparator(cmp: C) -> TreeMap<K, V, C> {
        TreeMap {
            root: None,
            length: 0,
            cmp: cmp,
        }
    }

    /// Returns the comparator according to which the `TreeMap` is ordered.
    pub fn comparator(&self) -> &C {
        &self.cmp
    }

    /// Gets a lazy iterator over the keys in the map, in ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// // Print "a", "b", "c" in order.
    /// for x in map.keys() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn keys<'a>(&'a self) -> Keys<'a, K, V> {
        fn first<A, B>((a, _): (A, B)) -> A {
            a
        }
        let first: fn((&'a K, &'a V)) -> &'a K = first; // coerce to fn pointer

        Keys(self.iter().map(first))
    }

    /// Gets a lazy iterator over the values in the map, in ascending order
    /// with respect to the corresponding keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// // Print 1, 2, 3 ordered by keys.
    /// for x in map.values() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn values<'a>(&'a self) -> Values<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B {
            b
        }
        let second: fn((&'a K, &'a V)) -> &'a V = second; // coerce to fn pointer

        Values(self.iter().map(second))
    }

    /// Gets a lazy iterator over the values in the map, in ascending order
    /// with respect to the corresponding keys, returning a mutable reference
    /// to each value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// for x in map.values_mut() {
    ///     *x += 1;
    /// }
    ///
    /// // Print 2, 3, 4 ordered by keys.
    /// for x in map.values() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn values_mut<'a>(&'a mut self) -> ValuesMut<'a, K, V> {
        fn second<A, B>((_, b): (A, B)) -> B {
            b
        }
        let second: fn((&'a K, &'a mut V)) -> &'a mut V = second; // coerce to fn pointer

        ValuesMut(self.iter_mut().map(second))
    }

    /// Gets a lazy iterator over the key-value pairs in the map, in ascending order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// // Print contents in ascending order
    /// for (key, value) in map.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<K, V, Forward> {
        Iter { iter_mut: (unsafe { &mut *(self as *const Self as *mut Self) }).iter_mut() }
    }

    fn iter_mut_dir<D: Direction>(&mut self) -> IterMut<K, V, D> {
        IterMut {
            stack: vec![],
            node: deref_mut(&mut self.root),
            direction: PhantomData,
        }
    }

    /// Gets a lazy forward iterator over the key-value pairs in the
    /// map, with the values being mutable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// // Add 10 until we find "b"
    /// for (key, value) in map.iter_mut() {
    ///     *value += 10;
    ///     if key == &"b" { break }
    /// }
    ///
    /// assert_eq!(map.get(&"a"), Some(&11));
    /// assert_eq!(map.get(&"b"), Some(&12));
    /// assert_eq!(map.get(&"c"), Some(&3));
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<K, V, Forward> {
        self.iter_mut_dir()
    }

    /// Gets a lazy iterator that consumes the treemap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    /// let mut map = TreeMap::new();
    /// map.insert("a", 1);
    /// map.insert("c", 3);
    /// map.insert("b", 2);
    ///
    /// // Not possible with a regular `.iter()`
    /// let vec: Vec<(&str, i32)> = map.into_iter().collect();
    /// assert_eq!(vec, vec![("a", 1), ("b", 2), ("c", 3)]);
    /// ```
    pub fn into_iter(self) -> IntoIter<K, V> {
        let TreeMap { root, length, .. } = self;
        let stk = match root {
            None => vec![],
            Some(b) => vec![*b],
        };
        IntoIter {
            stack: stk,
            remaining: length,
        }
    }

    /// Return the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return true if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, removing all values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut a = TreeMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.root = None;
        self.length = 0
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    #[inline]
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
        where C: Compare<Q, K>
    {
        // FIXME: redundant, but a bug in method-level where clauses requires it
        fn f<'r, K, V, C, Q: ?Sized>(node: &'r Option<Box<TreeNode<K, V>>>,
                                     cmp: &C,
                                     key: &Q)
                                     -> Option<&'r V>
            where C: Compare<Q, K>
        {
            tree_find_with(node, |k| cmp.compare(key, k))
        }

        f(&self.root, &self.cmp, key)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    #[inline]
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where C: Compare<Q, K>
    {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1, "a");
    /// match map.get_mut(&1) {
    ///     Some(x) => *x = "b",
    ///     None => (),
    /// }
    /// assert_eq!(map[&1], "b");
    /// ```
    #[inline]
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
        where C: Compare<Q, K>
    {
        // FIXME: redundant, but a bug in method-level where clauses requires it
        fn f<'r, K, V, C, Q: ?Sized>(node: &'r mut Option<Box<TreeNode<K, V>>>,
                                     cmp: &C,
                                     key: &Q)
                                     -> Option<&'r mut V>
            where C: Compare<Q, K>
        {
            tree_find_with_mut(node, |k| cmp.compare(key, k))
        }

        f(&mut self.root, &self.cmp, key)
    }

    /// Inserts a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map[&37], "c");
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut val = Some(value);
        let ret = self.get_or_insert(key, || val.take().unwrap());
        match val {
            None => None,
            Some(val) => Some(replace(ret, val)),
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but the ordering
    /// on the borrowed form *must* match the ordering on the key type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
        where C: Compare<Q, K>
    {
        let ret = remove(&mut self.root, key, &self.cmp);
        if ret.is_some() {
            self.length -= 1
        }
        ret
    }

    /// If a value for `key` does not exist, create one by callling `default`.
    /// Returns a mut reference to the new or existing value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// let mut count: TreeMap<&str, usize> = TreeMap::new();
    ///
    /// // count the number of occurrences of letters in the vec
    /// for x in vec!["a","b","a","c","a","b"] {
    ///     *count.get_or_insert(x, || 0) += 1;
    /// }
    /// assert_eq!(count[&"a"], 3);
    /// ```
    pub fn get_or_insert<F>(&mut self, key: K, default: F) -> &mut V
        where F: FnOnce() -> V
    {
        let (inserted, ret) = insert(&mut self.root, key, default, &self.cmp);
        self.length += inserted;
        unsafe { &mut *ret }
    }

    /// Returns the value for which `f(key)` returns `Equal`. `f` is invoked
    /// with current key and guides tree navigation. That means `f` should
    /// be aware of natural ordering of the tree.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use stable_bst::TreeMap;
    ///
    /// fn get_headers() -> TreeMap<&'static str, &'static str> {
    ///     let mut result = TreeMap::new();
    ///     result.insert("Content-Type", "application/xml");
    ///     result.insert("User-Agent", "Curl-Rust/0.1");
    ///     result
    /// }
    ///
    /// let headers = get_headers();
    /// let ua_key = "User-Agent";
    /// let ua = headers.find_with(|&k| {
    ///    ua_key.cmp(k)
    /// });
    ///
    /// assert_eq!(*ua.unwrap(), "Curl-Rust/0.1");
    /// ```
    #[inline]
    pub fn find_with<F>(&self, f: F) -> Option<&V>
        where F: FnMut(&K) -> Ordering
    {
        tree_find_with(&self.root, f)
    }

    /// Returns the value for which `f(key)` returns `Equal`. `f` is invoked
    /// with current key and guides tree navigation. That means `f` should
    /// be aware of natural ordering of the tree.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut t = stable_bst::TreeMap::new();
    /// t.insert("Content-Type", "application/xml");
    /// t.insert("User-Agent", "Curl-Rust/0.1");
    ///
    /// let new_ua = "Safari/156.0";
    /// match t.find_with_mut(|&k| "User-Agent".cmp(k)) {
    ///    Some(x) => *x = new_ua,
    ///    None => panic!(),
    /// }
    ///
    /// assert_eq!(t.get(&"User-Agent"), Some(&new_ua));
    /// ```
    #[inline]
    pub fn find_with_mut<F>(&mut self, f: F) -> Option<&mut V>
        where F: FnMut(&K) -> Ordering
    {
        tree_find_with_mut(&mut self.root, f)
    }
}

// range iterators.

impl<K, V, C> TreeMap<K, V, C>
    where C: Compare<K>
{
    fn compare_bound<D, Q: ?Sized>(&self, bound: Bound<&Q>, key: &K) -> Ordering
        where C: Compare<Q, K>,
              D: Direction
    {
        if D::forward() {
            match bound {
                Bound::Unbounded => Less,
                Bound::Included(k) => self.cmp.compare(k, key),
                Bound::Excluded(k) => {
                    match self.cmp.compare(k, key) {
                        Less => Less,
                        Greater | Equal => Greater,
                    }
                }
            }
        } else {
            match bound {
                Bound::Unbounded => Greater,
                Bound::Included(k) => self.cmp.compare(k, key),
                Bound::Excluded(k) => {
                    match self.cmp.compare(k, key) {
                        Less | Equal => Less,
                        Greater => Greater,
                    }
                }
            }
        }
    }

    fn bound_setup<'a, D, Q: ?Sized>(&self,
                                     mut iter: IterMut<'a, K, V, D>,
                                     bound: Bound<&Q>)
                                     -> IterMut<'a, K, V, D>
        where C: Compare<Q, K>,
              D: Direction
    {
        loop {
            if !iter.node.is_null() {
                let node_k = unsafe { &(*iter.node).key };
                match self.compare_bound::<D, Q>(bound, node_k) {
                    Less => iter.traverse_before(),
                    Greater => iter.traverse_after(),
                    Equal => {
                        iter.traverse_complete();
                        return iter;
                    }
                }
            } else {
                iter.traverse_complete();
                return iter;
            }
        }
    }

    /// Constructs a mutable double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use stable_bst::TreeMap;
    /// use stable_bst::Bound::{Included, Excluded};
    ///
    /// let mut map: TreeMap<&str, i32> = ["Alice", "Bob", "Carol", "Cheryl"].iter()
    ///                                                                      .map(|&s| (s, 0))
    ///                                                                      .collect();
    /// for (_, balance) in map.range_mut(Included(&"B"), Excluded(&"Cheryl")) {
    ///     *balance += 100;
    /// }
    /// for (name, balance) in &map {
    ///     println!("{} => {}", name, balance);
    /// }
    /// ```
    pub fn range_mut<'a, Min: ?Sized, Max: ?Sized>(&'a mut self,
                                                   min: Bound<&Min>,
                                                   max: Bound<&Max>)
                                                   -> RangeMut<'a, K, V>
        where C: Compare<Min, K> + Compare<Max, K>
    {
        let none = None;
        let mut node = &self.root;
        loop {
            match node {
                &None => {
                    return RangeMut {
                        start: (unsafe { &mut *(self as *const Self as *mut Self) }).iter_mut_dir(),
                        end: (unsafe { &mut *(self as *const Self as *mut Self) }).iter_mut_dir(),
                        empty: true,
                    }
                }
                &Some(ref n) => {
                    match (self.compare_bound::<Forward, Min>(min, &n.key),
                           self.compare_bound::<Backward, Max>(max, &n.key)) {
                        // If both endpoints are in the same subtree, descend into that subtree
                        (Less, Less) => node = &n.left,
                        (Greater, Greater) => node = &n.right,
                        // If start endpoint is actually > the end endpoint, return empty iterator
                        (Equal, Less) | (Greater, Less) | (Greater, Equal) => node = &none,
                        (Less, Equal) | (Less, Greater) | (Equal, Equal) | (Equal, Greater) => {
                            // We now know that the iterator will be non-empty.
                            // Populate the iterators.
                            let start: IterMut<K, V, Forward> = IterMut {
                                stack: vec![],
                                node: n.deref() as *const TreeNode<K, V> as *mut TreeNode<K, V>,
                                direction: PhantomData,
                            };
                            let end: IterMut<K, V, Backward> = IterMut {
                                stack: vec![],
                                node: n.deref() as *const TreeNode<K, V> as *mut TreeNode<K, V>,
                                direction: PhantomData,
                            };
                            return RangeMut {
                                start: self.bound_setup(start, min),
                                end: self.bound_setup(end, max),
                                empty: false,
                            };
                        }
                    };
                }
            }
        }
    }

    /// Constructs a double-ended iterator over a sub-range of elements in the map, starting
    /// at min, and ending at max. If min is `Unbounded`, then it will be treated as "negative
    /// infinity", and if max is `Unbounded`, then it will be treated as "positive infinity".
    /// Thus range(Unbounded, Unbounded) will yield the whole collection.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use stable_bst::TreeMap;
    /// use stable_bst::Bound::{Included, Unbounded};
    ///
    /// let mut map = TreeMap::new();
    /// map.insert(3, "a");
    /// map.insert(5, "b");
    /// map.insert(8, "c");
    /// for (&key, &value) in map.range(Included(&4), Included(&8)) {
    ///     println!("{}: {}", key, value);
    /// }
    /// assert_eq!(Some((&5, &"b")), map.range(Included(&4), Unbounded).next());
    /// ```
    pub fn range<'a, Min: ?Sized, Max: ?Sized>(&'a self,
                                               min: Bound<&Min>,
                                               max: Bound<&Max>)
                                               -> Range<'a, K, V>
        where C: Compare<Min, K> + Compare<Max, K>
    {
        Range { range: (unsafe { &mut *(self as *const Self as *mut Self) }).range_mut(min, max) }
    }
}

pub struct RangeMut<'a, K: 'a, V: 'a> {
    start: IterMut<'a, K, V, Forward>,
    end: IterMut<'a, K, V, Backward>,
    empty: bool,
}

pub struct Range<'a, K: 'a, V: 'a> {
    range: RangeMut<'a, K, V>,
}

pub trait Direction {
    fn forward() -> bool;
}

pub enum Forward {}

impl Direction for Forward {
    fn forward() -> bool {
        true
    }
}

pub enum Backward {}

impl Direction for Backward {
    fn forward() -> bool {
        false
    }
}

/// Lazy forward iterator over a map
pub struct Iter<'a, K: 'a, V: 'a, D: Direction> {
    iter_mut: IterMut<'a, K, V, D>,
}

/// Lazy forward iterator over a map that allows for the mutation of
/// the values.
pub struct IterMut<'a, K: 'a, V: 'a, D: Direction> {
    stack: Vec<&'a mut TreeNode<K, V>>,
    // Unfortunately, we require some unsafe-ness to get around the
    // fact that we would be storing a reference *into* one of the
    // nodes in the stack.
    //
    // As far as the compiler knows, this would let us invalidate the
    // reference by assigning a new value to this node's position in
    // its parent, which would cause this current one to be
    // deallocated so this reference would be invalid. (i.e. the
    // compilers complaints are 100% correct.)
    //
    // However, as far as you humans reading this code know (or are
    // about to know, if you haven't read far enough down yet), we are
    // only reading from the TreeNode.{left,right} fields. the only
    // thing that is ever mutated is the .value field (although any
    // actual mutation that happens is done externally, by the
    // iterator consumer). So, don't be so concerned, rustc, we've got
    // it under control.
    //
    // (This field can legitimately be null.)
    node: *mut TreeNode<K, V>,
    direction: PhantomData<D>,
}

/// TreeMap keys iterator.
pub struct Keys<'a, K: 'a, V: 'a>(iter::Map<Iter<'a, K, V, Forward>, fn((&'a K, &'a V)) -> &'a K>);

/// TreeMap values iterator.
pub struct Values<'a, K: 'a, V: 'a>(iter::Map<Iter<'a, K, V, Forward>,
                                              fn((&'a K, &'a V)) -> &'a V>);

/// TreeMap values iterator.
pub struct ValuesMut<'a, K: 'a, V: 'a>(iter::Map<IterMut<'a, K, V, Forward>,
                                                 fn((&'a K, &'a mut V)) -> &'a mut V>);

impl<'a, K, V, D: Direction> IterMut<'a, K, V, D> {
    #[inline(always)]
    fn next_(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.normalize();
        self.next_node()
    }

    fn normalize(&mut self) {
        while !self.node.is_null() {
            let node = unsafe { &mut *self.node };
            {
                let next_node = if D::forward() {
                    &mut node.left
                } else {
                    &mut node.right
                };
                self.node = deref_mut(next_node);
            }
            self.stack.push(node);
        }
    }

    fn next_node(&mut self) -> Option<(&'a K, &'a mut V)> {
        return self.stack.pop().map(|node| {
            let next_node = if D::forward() {
                &mut node.right
            } else {
                &mut node.left
            };
            self.node = deref_mut(next_node);
            (&node.key, &mut node.value)
        });
    }

    #[inline]
    fn traverse_before(&mut self) {
        let node = unsafe { &mut *self.node };
        self.node = deref_mut(&mut node.left);
        if D::forward() {
            self.stack.push(node);
        }
    }

    #[inline]
    fn traverse_after(&mut self) {
        let node = unsafe { &mut *self.node };
        self.node = deref_mut(&mut node.right);
        if !D::forward() {
            self.stack.push(node);
        }
    }

    #[inline]
    fn traverse_complete(&mut self) {
        if !self.node.is_null() {
            unsafe {
                self.stack.push(&mut *self.node);
            }
            self.node = ptr::null_mut();
        }
    }
}

impl<'a, K, V, D: Direction> Iterator for IterMut<'a, K, V, D> {
    type Item = (&'a K, &'a mut V);
    /// Advances the iterator to the next node (in order) and return a
    /// tuple with a reference to the key and value. If there are no
    /// more nodes, return `None`.
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        self.next_()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<'a, K, V> Iterator for RangeMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.empty {
            return None;
        }
        self.start.normalize();
        self.end.normalize();
        self.empty = match (self.start.stack.last(), self.end.stack.last()) {
            (None, _) | (_, None) => true,
            (Some(n1), Some(n2)) => *n1 as *const TreeNode<K, V> == *n2 as *const TreeNode<K, V>,
        };
        self.start.next_node()
    }
}

impl<'a, K, V> DoubleEndedIterator for RangeMut<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a mut V)> {
        if self.empty {
            return None;
        }
        self.start.normalize();
        self.end.normalize();
        self.empty = match (self.start.stack.last(), self.end.stack.last()) {
            (None, _) | (_, None) => true,
            (Some(n1), Some(n2)) => *n1 as *const TreeNode<K, V> == *n2 as *const TreeNode<K, V>,
        };
        self.end.next_node()
    }
}

impl<'a, K, V> Iterator for Range<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.range.next().map(|o| match o {
            (k, v) => (k, &*v),
        })
    }
}

impl<'a, K, V> DoubleEndedIterator for Range<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> {
        self.range.next_back().map(|o| match o {
            (k, v) => (k, &*v),
        })
    }
}

impl<'a, K, V, D: Direction> Iterator for Iter<'a, K, V, D> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> {
        self.iter_mut.next_().map(|o| match o {
            (k, v) => (k, &*v),
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

fn deref_mut<K, V>(x: &mut Option<Box<TreeNode<K, V>>>) -> *mut TreeNode<K, V> {
    match *x {
        Some(ref mut n) => &mut **n,
        None => ptr::null_mut(),
    }
}

/// Lazy forward iterator over a map that consumes the map while iterating
pub struct IntoIter<K, V> {
    stack: Vec<TreeNode<K, V>>,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        while !self.stack.is_empty() {
            let TreeNode { key, value, left, right, level } = self.stack.pop().unwrap();

            match left {
                Some(b_left) => {
                    let n = TreeNode {
                        key: key,
                        value: value,
                        left: None,
                        right: right,
                        level: level,
                    };
                    self.stack.push(n);
                    self.stack.push(*b_left);
                }
                None => {
                    match right {
                        Some(b_right) => self.stack.push(*b_right),
                        None => (),
                    }
                    self.remaining -= 1;
                    return Some((key, value));
                }
            }
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    #[inline]
    fn next(&mut self) -> Option<&'a K> {
        self.0.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    #[inline]
    fn next(&mut self) -> Option<&'a V> {
        self.0.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;
    #[inline]
    fn next(&mut self) -> Option<&'a mut V> {
        self.0.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

// Nodes keep track of their level in the tree, starting at 1 in the
// leaves and with a red child sharing the level of the parent.
#[derive(Clone)]
struct TreeNode<K, V> {
    key: K,
    value: V,
    left: Option<Box<TreeNode<K, V>>>,
    right: Option<Box<TreeNode<K, V>>>,
    level: usize,
}

impl<K, V> TreeNode<K, V> {
    /// Creates a new tree node.
    #[inline]
    pub fn new(key: K, value: V) -> TreeNode<K, V> {
        TreeNode {
            key: key,
            value: value,
            left: None,
            right: None,
            level: 1,
        }
    }
}

// Remove left horizontal link by rotating right
fn skew<K, V>(node: &mut Box<TreeNode<K, V>>) {
    if node.left.as_ref().map_or(false, |x| x.level == node.level) {
        let mut save = node.left.take().unwrap();
        swap(&mut node.left, &mut save.right); // save.right now None
        swap(node, &mut save);
        node.right = Some(save);
    }
}

// Remove dual horizontal link by rotating left and increasing level of
// the parent
fn split<K, V>(node: &mut Box<TreeNode<K, V>>) {
    if node.right.as_ref().map_or(false,
                                  |x| x.right.as_ref().map_or(false, |y| y.level == node.level)) {
        let mut save = node.right.take().unwrap();
        swap(&mut node.right, &mut save.left); // save.left now None
        save.level += 1;
        swap(node, &mut save);
        node.left = Some(save);
    }
}

// Next 2 functions have the same convention: comparator gets
// at input current key and returns search_key cmp cur_key
// (i.e. search_key.cmp(&cur_key))
fn tree_find_with<K, V, F>(node: &Option<Box<TreeNode<K, V>>>, mut f: F) -> Option<&V>
    where F: FnMut(&K) -> Ordering
{
    let mut current = node;
    loop {
        match *current {
            Some(ref r) => {
                match f(&r.key) {
                    Less => current = &r.left,
                    Greater => current = &r.right,
                    Equal => return Some(&r.value),
                }
            }
            None => return None,
        }
    }
}

// See comments above tree_find_with
fn tree_find_with_mut<K, V, F>(node: &mut Option<Box<TreeNode<K, V>>>, mut f: F) -> Option<&mut V>
    where F: FnMut(&K) -> Ordering
{

    let mut current = node;
    loop {
        let temp = current; // hack to appease borrowck
        match *temp {
            Some(ref mut r) => {
                match f(&r.key) {
                    Less => current = &mut r.left,
                    Greater => current = &mut r.right,
                    Equal => return Some(&mut r.value),
                }
            }
            None => return None,
        }
    }
}

fn insert<'a, K, V, F, C>(node: &'a mut Option<Box<TreeNode<K, V>>>,
                          key: K,
                          default: F,
                          cmp: &C)
                          -> (usize, *mut V)
    where C: Compare<K>,
          K: 'a,
          V: 'a,
          F: FnOnce() -> V
{

    match *node {
        Some(ref mut save) => {
            match cmp.compare(&key, &save.key) {
                Less => {
                    let ret = insert(&mut save.left, key, default, cmp);
                    skew(save);
                    split(save);
                    ret
                }
                Greater => {
                    let ret = insert(&mut save.right, key, default, cmp);
                    skew(save);
                    split(save);
                    ret
                }
                Equal => (0, &mut save.value),
            }
        }
        None => {
            *node = Some(Box::new(TreeNode::new(key, default())));
            (1, &mut node.as_mut().unwrap().value)
        }
    }
}

fn remove<K, V, C, Q: ?Sized>(node: &mut Option<Box<TreeNode<K, V>>>, key: &Q, cmp: &C) -> Option<V>
    where C: Compare<Q, K>
{

    fn heir_swap<K, V>(node: &mut Box<TreeNode<K, V>>, child: &mut Option<Box<TreeNode<K, V>>>) {
        // *could* be done without recursion, but it won't borrow check
        for x in child.iter_mut() {
            if x.right.is_some() {
                heir_swap(node, &mut x.right);
            } else {
                swap(&mut node.key, &mut x.key);
                swap(&mut node.value, &mut x.value);
            }
        }
    }

    match *node {
        None => {
            return None; // bottom of tree
        }
        Some(ref mut save) => {
            let (ret, rebalance) = match cmp.compare(key, &save.key) {
                Less => (remove(&mut save.left, key, cmp), true),
                Greater => (remove(&mut save.right, key, cmp), true),
                Equal => {
                    if save.left.is_some() {
                        if save.right.is_some() {
                            let mut left = save.left.take().unwrap();
                            if left.right.is_some() {
                                heir_swap(save, &mut left.right);
                            } else {
                                swap(&mut save.key, &mut left.key);
                                swap(&mut save.value, &mut left.value);
                            }
                            save.left = Some(left);
                            (remove(&mut save.left, key, cmp), true)
                        } else {
                            let new = save.left.take().unwrap();
                            let TreeNode { value, .. } = *replace(save, new);
                            *save = save.left.take().unwrap();
                            (Some(value), true)
                        }
                    } else if save.right.is_some() {
                        let new = save.right.take().unwrap();
                        let TreeNode { value, .. } = *replace(save, new);
                        (Some(value), true)
                    } else {
                        (None, false)
                    }
                }
            };

            if rebalance {
                let left_level = save.left.as_ref().map_or(0, |x| x.level);
                let right_level = save.right.as_ref().map_or(0, |x| x.level);

                // re-balance, if necessary
                if left_level < save.level - 1 || right_level < save.level - 1 {
                    save.level -= 1;

                    if right_level > save.level {
                        let save_level = save.level;
                        for x in save.right.iter_mut() {
                            x.level = save_level
                        }
                    }

                    skew(save);

                    for right in save.right.iter_mut() {
                        skew(right);
                        for x in right.right.iter_mut() {
                            skew(x)
                        }
                    }

                    split(save);
                    for x in save.right.iter_mut() {
                        split(x)
                    }
                }

                return ret;
            }
        }
    }
    return match node.take() {
        Some(b) => {
            let TreeNode { value, .. } = *b;
            Some(value)
        }
        None => panic!(),
    };
}

impl<K, V, C> iter::FromIterator<(K, V)> for TreeMap<K, V, C>
    where C: Compare<K> + Default
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> TreeMap<K, V, C> {
        let mut map: TreeMap<K, V, C> = Default::default();
        map.extend(iter);
        map
    }
}

impl<K, V, C> Extend<(K, V)> for TreeMap<K, V, C>
    where C: Compare<K>
{
    #[inline]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Hash, V: Hash, C> Hash for TreeMap<K, V, C>
    where C: Compare<K>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}

impl<'a, K, V, C> IntoIterator for &'a TreeMap<K, V, C>
    where C: Compare<K>
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, Forward>;
    fn into_iter(self) -> Iter<'a, K, V, Forward> {
        self.iter()
    }
}

impl<'a, K, V, C> IntoIterator for &'a mut TreeMap<K, V, C>
    where C: Compare<K>
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V, Forward>;
    fn into_iter(self) -> IterMut<'a, K, V, Forward> {
        self.iter_mut()
    }
}

impl<K, V, C> IntoIterator for TreeMap<K, V, C>
    where C: Compare<K>
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> IntoIter<K, V> {
        self.into_iter()
    }
}

#[cfg(feature="ordered_iter")]
impl<'a, K, V> ::ordered_iter::OrderedMapIterator for Iter<'a, K, V, Forward> {
    type Key = &'a K;
    type Val = &'a V;
}

#[cfg(test)]
mod test_treemap {
    use rand::{self, Rng};

    use super::{TreeMap, TreeNode, Range, RangeMut};
    use super::super::Bound;

    #[test]
    fn find_empty() {
        let m: TreeMap<i32, i32> = TreeMap::new();
        assert!(m.get(&5) == None);
    }

    #[test]
    fn find_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(5, 3).is_none());
        assert!(m.insert(9, 3).is_none());
        assert_eq!(m.get(&2), None);
    }

    #[test]
    fn find_with_empty() {
        let m: TreeMap<&'static str, i32> = TreeMap::new();
        assert!(m.find_with(|&k| "test".cmp(k)) == None);
    }

    #[test]
    fn find_with_not_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2).is_none());
        assert!(m.insert("test2", 3).is_none());
        assert!(m.insert("test3", 3).is_none());
        assert_eq!(m.find_with(|&k| "test4".cmp(k)), None);
    }

    #[test]
    fn find_with_found() {
        let mut m = TreeMap::new();
        assert!(m.insert("test1", 2).is_none());
        assert!(m.insert("test2", 3).is_none());
        assert!(m.insert("test3", 4).is_none());
        assert_eq!(m.find_with(|&k| "test2".cmp(k)), Some(&3));
    }

    #[test]
    fn test_find_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(&5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(&5), Some(&new));
    }

    #[test]
    fn test_find_with_mut() {
        let mut m = TreeMap::new();
        assert!(m.insert("t1", 12).is_none());
        assert!(m.insert("t2", 8).is_none());
        assert!(m.insert("t5", 14).is_none());
        let new = 100;

        match m.find_with_mut(|&k| "t5".cmp(k)) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.find_with(|&k| "t5".cmp(k)), Some(&new));
    }

    #[test]
    fn insert_replace() {
        let mut m = TreeMap::new();
        assert!(m.insert(5, 2).is_none());
        assert!(m.insert(2, 9).is_none());
        assert!(!m.insert(2, 11).is_none());
        assert_eq!(m.get(&2).unwrap(), &11);
    }

    #[test]
    fn test_get_or_insert() {
        let mut m = TreeMap::new();
        assert_eq!(*m.get_or_insert(5, || 2), 2);
        assert_eq!(*m.get_or_insert(2, || 9), 9);
        assert_eq!(*m.get_or_insert(2, || 7), 9);
        *m.get_or_insert(2, || 7) = 8;
        assert_eq!(*m.get(&2).unwrap(), 8);
    }

    #[test]
    fn test_clear() {
        let mut m = TreeMap::new();
        m.clear();
        assert!(m.insert(5, 11).is_none());
        assert!(m.insert(12, -3).is_none());
        assert!(m.insert(19, 2).is_none());
        m.clear();
        assert!(m.get(&5).is_none());
        assert!(m.get(&12).is_none());
        assert!(m.get(&19).is_none());
        assert!(m.is_empty());
    }

    #[test]
    fn u8_map() {
        let mut m = TreeMap::new();

        let k1 = "foo".as_bytes();
        let k2 = "bar".as_bytes();
        let v1 = "baz".as_bytes();
        let v2 = "foobar".as_bytes();

        m.insert(k1.clone(), v1.clone());
        m.insert(k2.clone(), v2.clone());

        assert_eq!(m.get(&k2), Some(&v2));
        assert_eq!(m.get(&k1), Some(&v1));
    }

    fn check_equal<K: PartialEq + Ord, V: PartialEq>(ctrl: &[(K, V)], map: &TreeMap<K, V>) {
        assert_eq!(ctrl.is_empty(), map.is_empty());
        for x in ctrl.iter() {
            let &(ref k, ref v) = x;
            assert!(map.get(k).unwrap() == v)
        }
        for (map_k, map_v) in map.iter() {
            let mut found = false;
            for x in ctrl.iter() {
                let &(ref ctrl_k, ref ctrl_v) = x;
                if *map_k == *ctrl_k {
                    assert!(*map_v == *ctrl_v);
                    found = true;
                    break;
                }
            }
            assert!(found);
        }
    }

    fn check_left<K: Ord, V>(node: &Option<Box<TreeNode<K, V>>>, parent: &Box<TreeNode<K, V>>) {
        match *node {
            Some(ref r) => {
                assert_eq!(r.key.cmp(&parent.key), ::std::cmp::Ordering::Less);
                assert!(r.level == parent.level - 1); // left is black
                check_left(&r.left, r);
                check_right(&r.right, r, false);
            }
            None => assert!(parent.level == 1), // parent is leaf
        }
    }

    fn check_right<K: Ord, V>(node: &Option<Box<TreeNode<K, V>>>,
                              parent: &Box<TreeNode<K, V>>,
                              parent_red: bool) {
        match *node {
            Some(ref r) => {
                assert_eq!(r.key.cmp(&parent.key), ::std::cmp::Ordering::Greater);
                let red = r.level == parent.level;
                if parent_red {
                    assert!(!red)
                } // no dual horizontal links
            // Right red or black
                assert!(red || r.level == parent.level - 1);
                check_left(&r.left, r);
                check_right(&r.right, r, red);
            }
            None => assert!(parent.level == 1), // parent is leaf
        }
    }

    fn check_structure<K: Ord, V>(map: &TreeMap<K, V>) {
        match map.root {
            Some(ref r) => {
                check_left(&r.left, r);
                check_right(&r.right, r, false);
            }
            None => (),
        }
    }

    #[test]
    fn test_rand_int() {
        let mut map: TreeMap<i32, i32> = TreeMap::new();
        let mut ctrl = vec![];

        check_equal(&ctrl, &map);
        assert!(map.get(&5).is_none());

        let seed: &[_] = &[42];
        let mut rng: rand::IsaacRng = rand::SeedableRng::from_seed(seed);

        for _ in 0..3 {
            for _ in 0..90 {
                let k = rng.gen();
                let v = rng.gen();
                if !ctrl.iter().any(|x| x == &(k, v)) {
                    assert!(map.insert(k, v).is_none());
                    ctrl.push((k, v));
                    check_structure(&map);
                    check_equal(&ctrl, &map);
                }
            }

            for _ in 0..30 {
                let r = rng.gen_range(0, ctrl.len());
                let (key, _) = ctrl.remove(r);
                assert!(map.remove(&key).is_some());
                check_structure(&map);
                check_equal(&ctrl, &map);
            }
        }
    }

    #[test]
    fn test_len() {
        let mut m = TreeMap::new();
        assert!(m.insert(3, 6).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(0, 0).is_none());
        assert_eq!(m.len(), 2);
        assert!(m.insert(4, 8).is_none());
        assert_eq!(m.len(), 3);
        assert!(m.remove(&3).is_some());
        assert_eq!(m.len(), 2);
        assert!(!m.remove(&5).is_some());
        assert_eq!(m.len(), 2);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 3);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn test_iterator() {
        let mut m = TreeMap::new();

        assert!(m.insert(3, 6).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

        let mut n = 0;
        for (k, v) in m.iter() {
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
            n += 1;
        }
        assert_eq!(n, 5);
    }

    #[test]
    fn test_interval_iteration() {
        let mut m = TreeMap::new();
        for i in 1..100 {
            assert!(m.insert(i * 2, i * 4).is_none());
        }

        for i in 1..198 {
            let mut lb_it = m.range(Bound::Included(&i), Bound::Unbounded);
            let (&k, &v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            assert_eq!(lb * 2, v);

            let mut ub_it = m.range(Bound::Excluded(&i), Bound::Unbounded);
            let (&k, &v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            assert_eq!(ub * 2, v);
        }
        let mut end_it = m.range(Bound::Included(&199), Bound::Unbounded);
        assert_eq!(end_it.next(), None);
    }

    #[test]
    fn test_mut_iter() {
        let mut m = TreeMap::new();
        for i in 0..10 {
            assert!(m.insert(i, 100 * i).is_none());
        }

        for (i, (&k, v)) in m.iter_mut().enumerate() {
            *v += k * 10 + i; // 000 + 00 + 0, 100 + 10 + 1, ...
        }

        for (&k, &v) in m.iter() {
            assert_eq!(v, 111 * k);
        }
    }

    #[test]
    fn test_mut_interval_iter() {
        let mut m_lower = TreeMap::new();
        let mut m_upper = TreeMap::new();
        for i in 1..100 {
            assert!(m_lower.insert(i * 2, i * 4).is_none());
            assert!(m_upper.insert(i * 2, i * 4).is_none());
        }

        for i in 1..199 {
            let mut lb_it = m_lower.range_mut(Bound::Included(&i), Bound::Unbounded);
            let (&k, v) = lb_it.next().unwrap();
            let lb = i + i % 2;
            assert_eq!(lb, k);
            *v -= k;
        }
        for i in 0..198 {
            let mut ub_it = m_upper.range_mut(Bound::Excluded(&i), Bound::Unbounded);
            let (&k, v) = ub_it.next().unwrap();
            let ub = i + 2 - i % 2;
            assert_eq!(ub, k);
            *v -= k;
        }

        assert!(m_lower.range_mut(Bound::Included(&199), Bound::Unbounded).next().is_none());

        assert!(m_upper.range_mut(Bound::Excluded(&199), Bound::Unbounded).next().is_none());

        assert!(m_lower.iter().all(|(_, &x)| x == 0));
        assert!(m_upper.iter().all(|(_, &x)| x == 0));
    }

    fn to_vec<K, V: Clone>(range: Range<K, V>) -> Vec<V> {
        range.map(|o| o.1.clone()).collect::<Vec<V>>()
    }

    fn skip<K, V>(mut range: Range<K, V>, mut f: u32, mut b: u32) -> Range<K, V> {
        while f > 0 {
            range.next();
            f -= 1;
        }
        while b > 0 {
            range.next_back();
            b -= 1;
        }
        range
    }

    #[test]
    fn test_range() {
        let mut m = TreeMap::new();
        for i in 0..10 {
            assert!(m.insert(i * 10, 100 * i).is_none());
        }

        assert_eq!(to_vec(m.range(Bound::Unbounded, Bound::Unbounded)),
                   vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900]);
        assert_eq!(to_vec(m.range(Bound::Included(&50), Bound::Unbounded)),
                   vec![500, 600, 700, 800, 900]);
        assert_eq!(to_vec(m.range(Bound::Excluded(&50), Bound::Unbounded)),
                   vec![600, 700, 800, 900]);

        assert_eq!(to_vec(m.range(Bound::Unbounded, Bound::Included(&70))),
                   vec![0, 100, 200, 300, 400, 500, 600, 700]);
        assert_eq!(to_vec(m.range(Bound::Unbounded, Bound::Excluded(&70))),
                   vec![0, 100, 200, 300, 400, 500, 600]);

        assert_eq!(to_vec(m.range(Bound::Included(&70), Bound::Included(&70))),
                   vec![700]);

        assert_eq!(to_vec(m.range(Bound::Included(&60), Bound::Excluded(&50))),
                   vec![]);
        assert_eq!(to_vec(m.range(Bound::Unbounded, Bound::Excluded(&0))),
                   vec![]);

        assert_eq!(m.range(Bound::Unbounded, Bound::Unbounded).next_back(),
                   Some((&90, &900)));
        assert_eq!(m.range(Bound::Unbounded, Bound::Included(&60)).next_back(),
                   Some((&60, &600)));
        assert_eq!(m.range(Bound::Unbounded, Bound::Excluded(&60)).next_back(),
                   Some((&50, &500)));

        assert_eq!(to_vec(skip(m.range(Bound::Unbounded, Bound::Unbounded), 3, 3)),
                   vec![300, 400, 500, 600]);
        assert_eq!(to_vec(skip(m.range(Bound::Unbounded, Bound::Unbounded), 5, 5)),
                   vec![]);
        assert_eq!(to_vec(skip(m.range(Bound::Unbounded, Bound::Unbounded), 3, 7)),
                   vec![]);
        assert_eq!(to_vec(skip(m.range(Bound::Unbounded, Bound::Unbounded), 7, 3)),
                   vec![]);
        assert_eq!(to_vec(skip(m.range(Bound::Unbounded, Bound::Unbounded), 8, 3)),
                   vec![]);
    }

    fn to_vec_mut<K, V: Clone>(range: RangeMut<K, V>) -> Vec<V> {
        range.map(|o| o.1.clone()).collect::<Vec<V>>()
    }

    fn skip_mut<K, V>(mut range: RangeMut<K, V>, mut f: u32, mut b: u32) -> RangeMut<K, V> {
        while f > 0 {
            range.next();
            f -= 1;
        }
        while b > 0 {
            range.next_back();
            b -= 1;
        }
        range
    }

    #[test]
    fn test_range_mut() {
        let mut m = TreeMap::new();
        for i in 0..10 {
            assert!(m.insert(i * 10, 20 * i).is_none());
        }
        for i in m.range_mut(Bound::Unbounded, Bound::Unbounded) {
            match i {
                (_, v) => *v *= 5,
            }
        }

        assert_eq!(to_vec_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded)),
                   vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900]);
        assert_eq!(to_vec_mut(m.range_mut(Bound::Included(&50), Bound::Unbounded)),
                   vec![500, 600, 700, 800, 900]);
        assert_eq!(to_vec_mut(m.range_mut(Bound::Excluded(&50), Bound::Unbounded)),
                   vec![600, 700, 800, 900]);

        assert_eq!(to_vec_mut(m.range_mut(Bound::Unbounded, Bound::Included(&70))),
                   vec![0, 100, 200, 300, 400, 500, 600, 700]);
        assert_eq!(to_vec_mut(m.range_mut(Bound::Unbounded, Bound::Excluded(&70))),
                   vec![0, 100, 200, 300, 400, 500, 600]);

        assert_eq!(to_vec_mut(m.range_mut(Bound::Included(&70), Bound::Included(&70))),
                   vec![700]);

        assert_eq!(to_vec_mut(m.range_mut(Bound::Included(&60), Bound::Excluded(&50))),
                   vec![]);
        assert_eq!(to_vec_mut(m.range_mut(Bound::Unbounded, Bound::Excluded(&0))),
                   vec![]);

        assert_eq!(m.range_mut(Bound::Unbounded, Bound::Unbounded).next_back(),
                   Some((&90, &mut 900)));
        assert_eq!(m.range_mut(Bound::Unbounded, Bound::Included(&60)).next_back(),
                   Some((&60, &mut 600)));
        assert_eq!(m.range_mut(Bound::Unbounded, Bound::Excluded(&60)).next_back(),
                   Some((&50, &mut 500)));

        assert_eq!(to_vec_mut(skip_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded), 3, 3)),
                   vec![300, 400, 500, 600]);
        assert_eq!(to_vec_mut(skip_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded), 5, 5)),
                   vec![]);
        assert_eq!(to_vec_mut(skip_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded), 3, 7)),
                   vec![]);
        assert_eq!(to_vec_mut(skip_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded), 7, 3)),
                   vec![]);
        assert_eq!(to_vec_mut(skip_mut(m.range_mut(Bound::Unbounded, Bound::Unbounded), 8, 3)),
                   vec![]);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: TreeMap<i32, char> = vec.into_iter().collect();
        let keys: Vec<i32> = map.keys().map(|&k| k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map = vec.into_iter().collect::<TreeMap<i32, char>>();
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values, vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_values_mut() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let mut map = vec.into_iter().collect::<TreeMap<i32, char>>();
        for ch in map.values_mut() {
            *ch = 'x';
        }
        let values = map.values().map(|&v| v).collect::<Vec<char>>();
        assert_eq!(values, vec!['x', 'x', 'x']);
    }

    #[test]
    fn test_eq() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a == b);
        assert!(a.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(0, 4).is_none());
        assert!(a != b);
        assert!(a.insert(5, 19).is_none());
        assert!(a != b);
        assert!(!b.insert(0, 5).is_none());
        assert!(a != b);
        assert!(b.insert(5, 19).is_none());
        assert!(a == b);
    }

    #[test]
    fn test_lt() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(!(a < b) && !(b < a));
        assert!(b.insert(0, 5).is_none());
        assert!(a < b);
        assert!(a.insert(0, 7).is_none());
        assert!(!(a < b) && b < a);
        assert!(b.insert(-2, 0).is_none());
        assert!(b < a);
        assert!(a.insert(-5, 2).is_none());
        assert!(a < b);
        assert!(a.insert(6, 2).is_none());
        assert!(a < b && !(b < a));
    }

    #[test]
    fn test_ord() {
        let mut a = TreeMap::new();
        let mut b = TreeMap::new();

        assert!(a <= b && a >= b);
        assert!(a.insert(1, 1).is_none());
        assert!(a > b && a >= b);
        assert!(b < a && b <= a);
        assert!(b.insert(2, 2).is_none());
        assert!(b > a && b >= a);
        assert!(a < b && a <= b);
    }

    #[test]
    fn test_debug() {
        let mut map = TreeMap::new();
        let empty: TreeMap<i32, i32> = TreeMap::new();

        map.insert(1, 2);
        map.insert(3, 4);

        assert_eq!(format!("{:?}", map), "{1: 2, 3: 4}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_lazy_iterator() {
        let mut m = TreeMap::new();
        let (x1, y1) = (2, 5);
        let (x2, y2) = (9, 12);
        let (x3, y3) = (20, -3);
        let (x4, y4) = (29, 5);
        let (x5, y5) = (103, 3);

        assert!(m.insert(x1, y1).is_none());
        assert!(m.insert(x2, y2).is_none());
        assert!(m.insert(x3, y3).is_none());
        assert!(m.insert(x4, y4).is_none());
        assert!(m.insert(x5, y5).is_none());

        let m = m;
        let mut a = m.iter();

        assert_eq!(a.next().unwrap(), (&x1, &y1));
        assert_eq!(a.next().unwrap(), (&x2, &y2));
        assert_eq!(a.next().unwrap(), (&x3, &y3));
        assert_eq!(a.next().unwrap(), (&x4, &y4));
        assert_eq!(a.next().unwrap(), (&x5, &y5));

        assert!(a.next().is_none());

        let mut b = m.iter();

        let expected = [(&x1, &y1), (&x2, &y2), (&x3, &y3), (&x4, &y4), (&x5, &y5)];
        let mut i = 0;

        for x in b.by_ref() {
            assert_eq!(expected[i], x);
            i += 1;

            if i == 2 {
                break;
            }
        }

        for x in b {
            assert_eq!(expected[i], x);
            i += 1;
        }
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: TreeMap<i32, i32> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.get(&k), Some(&v));
        }
    }

    #[test]
    fn test_index() {
        let mut map: TreeMap<i32, i32> = TreeMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[&2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map: TreeMap<i32, i32> = TreeMap::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[&4];
    }

    #[test]
    fn test_swap() {
        let mut m = TreeMap::new();
        assert_eq!(m.insert(1, 2), None);
        assert_eq!(m.insert(1, 3), Some(2));
        assert_eq!(m.insert(1, 4), Some(3));
    }

    #[test]
    fn test_pop() {
        let mut m = TreeMap::new();
        m.insert(1, 2);
        assert_eq!(m.remove(&1), Some(2));
        assert_eq!(m.remove(&1), None);
    }

    #[test]
    fn test_comparator_iterator() {
        use compare::{Compare, natural};

        let mut m = TreeMap::with_comparator(natural().rev());

        assert!(m.insert(3, 6).is_none());
        assert!(m.insert(0, 0).is_none());
        assert!(m.insert(4, 8).is_none());
        assert!(m.insert(2, 4).is_none());
        assert!(m.insert(1, 2).is_none());

        let mut n = 5;
        for (k, v) in m.iter() {
            n -= 1;
            assert_eq!(*k, n);
            assert_eq!(*v, n * 2);
        }
        assert_eq!(n, 0);
    }

    #[test]
    fn test_comparator_borrowed() {
        use compare::{Compare, natural};

        let mut m = TreeMap::with_comparator(natural().borrowing());

        assert!(m.insert("a".to_string(), 1).is_none());

        assert!(m.contains_key("a"));
        assert!(m.contains_key(&"a"));
        assert!(m.contains_key(&"a".to_string()));

        assert_eq!(m.get("a"), Some(&1));
        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m.get(&"a".to_string()), Some(&1));

        m["a"] = 2;

        assert_eq!(m["a"], 2);
        assert_eq!(m[&"a".to_string()], 2);

        m[&"a".to_string()] = 3;

        assert_eq!(m.remove("a"), Some(3));
        assert!(m.remove(&"a").is_none());
        assert!(m.remove(&"a".to_string()).is_none());
    }
}

#[cfg(all(test, feature="bench"))]
mod bench {
    use rand::{weak_rng, Rng};
    use test::{Bencher, black_box};

    use super::TreeMap;

    map_insert_rand_bench!{insert_rand_100,    100,    TreeMap}
    map_insert_rand_bench!{insert_rand_10_000, 10_000, TreeMap}

    map_insert_seq_bench!{insert_seq_100,    100,    TreeMap}
    map_insert_seq_bench!{insert_seq_10_000, 10_000, TreeMap}

    map_find_rand_bench!{find_rand_100,    100,    TreeMap}
    map_find_rand_bench!{find_rand_10_000, 10_000, TreeMap}

    map_find_seq_bench!{find_seq_100,    100,    TreeMap}
    map_find_seq_bench!{find_seq_10_000, 10_000, TreeMap}

    fn bench_iter(b: &mut Bencher, size: usize) {
        let mut map = TreeMap::<u32, u32>::new();
        let mut rng = weak_rng();

        for _ in 0..size {
            map.insert(rng.gen(), rng.gen());
        }

        b.iter(|| {
            for entry in map.iter() {
                black_box(entry);
            }
        });
    }

    #[bench]
    pub fn iter_20(b: &mut Bencher) {
        bench_iter(b, 20);
    }

    #[bench]
    pub fn iter_1000(b: &mut Bencher) {
        bench_iter(b, 1000);
    }

    #[bench]
    pub fn iter_100000(b: &mut Bencher) {
        bench_iter(b, 100000);
    }
}
