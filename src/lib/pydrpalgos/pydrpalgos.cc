#include "DrpAlgos/drp/Version.hh"
#include "DrpAlgos/host/AlgorithmHost.hh"

#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_drpalgos, drpalgos_module, py::mod_gil_not_used()) {
  py::classh<DrpAlgos::Version>(drpalgos_module, "Version")
    .def(py::init<>())
    .def(py::init<unsigned, unsigned, unsigned>())
    .def(py::init([](const std::string& s) {
      return DrpAlgos::Version::parse(s);
    }))
    .def("toString", &DrpAlgos::Version::toString)
    .def_readwrite("major", &DrpAlgos::Version::major)
    .def_readwrite("minor", &DrpAlgos::Version::minor)
    .def_readwrite("micro", &DrpAlgos::Version::micro);

  py::classh<DrpAlgos::HostConfig>(drpalgos_module, "HostConfig")
    .def(py::init<>())
    .def(py::init<unsigned, unsigned>())
    .def_readwrite("slots", &DrpAlgos::HostConfig::slots)
    .def_readwrite("inElems", &DrpAlgos::HostConfig::inElems);

  py::classh<DrpAlgos::BenchResult>(drpalgos_module, "BenchResult")
    .def(py::init<>())
    .def(py::init<std::size_t, float, float, float>())
    .def_readwrite("iterations", &DrpAlgos::BenchResult::iterations)
    .def_readwrite("meanUs", &DrpAlgos::BenchResult::meanUs)
    .def_readwrite("minUs", &DrpAlgos::BenchResult::minUs)
    .def_readwrite("maxUs", &DrpAlgos::BenchResult::maxUs);


  py::classh<DrpAlgos::AlgorithmHost>(drpalgos_module, "AlgorithmHost")
    .def(py::init<
           const std::string&,
           const DrpAlgos::Version&,
           const std::string&,
           const nlohmann::json&,
           DrpAlgos::HostConfig
         >())
    .def("payloadSize", &DrpAlgos::AlgorithmHost::payloadSize)
    .def("variableSize", &DrpAlgos::AlgorithmHost::variableSize)
    //.def("process",
    //     &DrpAlgos::AlgorithmHost::process,
    //     py::arg("frame")) // Need to convert to NumPy array
    .def("benchmark",
         &DrpAlgos::AlgorithmHost::benchmark,
         py::arg("iterations"));
}
