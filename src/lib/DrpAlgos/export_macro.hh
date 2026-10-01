#ifndef DRP_ALGOS_EXPORT_MACRO_HH
#define DRP_ALGOS_EXPORT_MACRO_HH

/**
 * @def DRP_ALGOS_API
 *
 * By default, DrpAlgos builds all symbols as hidden. Any public API must be
 * explicitly marked as so. This macro is a simple helper.
 *
 * @example class DRP_ALGOS_API PublicAPIClass {};
 */
#define DRP_ALGOS_API __attribute__((visibility("default")))

#endif // DRP_ALGOS_EXPORT_MACRO_HH
