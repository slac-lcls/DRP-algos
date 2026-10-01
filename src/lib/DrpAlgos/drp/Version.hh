#ifndef DRP_ALGOS_DRP_VERSION_HH
#define DRP_ALGOS_DRP_VERSION_HH

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

namespace DrpAlgos {
  /**
   * A semantic version specifier with major, minor, and micro components.
   */
  struct Version {
    unsigned major { 0 };
    unsigned minor { 0 };
    unsigned micro { 0 };

    /**
     * Parse a version string into a normalized Version.
     *
     * Accepts variants optionally prefixed with `v`. The major version is required,
     * but if the minor or micro version are ommitted they will be converted to 0
     * automatically. E.g. `v1` = 1.0.0, as do `1` and `1.0`.
     *
     * @param[in] s The version in string form.
     * @returns The parsed version with major, minor, and micro components.
     * @throws invalid_argument if the string not of form [v]MAJOR[.MINOR[.MICRO]].
     */
    static Version parse(const std::string& s) {
      auto fail = [&s]() {
        throw std::invalid_argument("Invalid version '" + s + "' (expected [v]MAJOR[.MINOR[.MICRO]])");
      };

      Version v;
      unsigned* parts[3] = { &v.major, &v.minor, &v.micro };
      std::size_t pos { (!s.empty() && (s[0] == 'v' || s[0] == 'V')) ? 1u : 0u };
      int n { 0 };

      while (true) {
        if (n == 3          ||
            pos >= s.size() ||
            !std::isdigit(static_cast<unsigned char>(s[pos]))) {
          fail();
        }

        unsigned long long val { 0 };
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
          val = val * 10 + static_cast<unsigned>(s[pos] - '0');

          if (val > UINT_MAX) {
            fail();
          }

          ++pos;
        }

        *parts[n++] = static_cast<unsigned>(val);

        if (pos == s.size()) {
          break;
        }

        if (s[pos] != '.') {
          fail();
        }

        ++pos;
      }

      return v;
    }

    /**
     * Convert a Version back into a string, with dot-delimited components.
     *
     * @returns The version as a string in format: "MAJOR.MINOR.MICRO".
     */
    std::string toString() const {
      return
        std::to_string(major) + "." +
        std::to_string(minor) + "." +
        std::to_string(micro);
    }

    /**
     * Whether this Version is less than the provided Version.
     *
     * @param[in] o The Version to compare to.
     * @returns Whether this Version is less than the provided Version.
     */
    bool operator<(const Version& o) const {
      return std::tie(major, minor, micro) < std::tie(o.major, o.minor, o.micro);
    }

    /**
     * Whether this Version is equal to the provided Version.
     *
     * @param[in] o The Version to compare to.
     * @returns Whether this Version is equal to the provided Version.
     */
    bool operator==(const Version& o) const {
      return std::tie(major, minor, micro) == std::tie(o.major, o.minor, o.micro);
    }
  };
} // namespace DrpAlgos

#endif // DRP_ALGOS_DRP_VERSION_HH
