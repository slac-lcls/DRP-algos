#include "DrpAlgos/drp/Version.hh"
#include "DrpAlgos/host/AlgorithmHost.hh"

#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
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

  py::classh<DrpAlgos::EventResult>(drpalgos_module, "EventResult")
    .def_property_readonly("payload",
                           [](const DrpAlgos::EventResult& self) {
                             return py::array_t<std::uint8_t>(static_cast<py::ssize_t>(self.payload.size()),
                                                              self.payload.data());
                           })
    .def_readonly("deviceMs", &DrpAlgos::EventResult::deviceMs);

  py::classh<DrpAlgos::BenchResult>(drpalgos_module, "BenchResult")
    .def(py::init<>())
    .def(py::init<std::size_t, float, float, float>())
    .def_readwrite("iterations", &DrpAlgos::BenchResult::iterations)
    .def_readwrite("meanUs", &DrpAlgos::BenchResult::meanUs)
    .def_readwrite("minUs", &DrpAlgos::BenchResult::minUs)
    .def_readwrite("maxUs", &DrpAlgos::BenchResult::maxUs);


  py::classh<DrpAlgos::AlgorithmHost>(drpalgos_module, "AlgorithmHost")
    .def(py::init([](const std::string& name,
                     const std::string& version,
                     const std::string& soname,
                     const std::string params,
                     DrpAlgos::HostConfig cfg) {
      return std::make_unique<DrpAlgos::AlgorithmHost>(name,
                                                       DrpAlgos::Version::parse(version),
                                                       soname,
                                                       nlohmann::json::parse(params),
                                                       cfg);
    }),
         py::arg("name"),
         py::arg("version"),
         py::arg("soname"),
         py::arg("params"),
         py::arg("cfg"))
    .def(py::init([](const std::string& name,
                     const DrpAlgos::Version& version,
                     const std::string& soname,
                     const std::string params,
                     DrpAlgos::HostConfig cfg) {
      return std::make_unique<DrpAlgos::AlgorithmHost>(name,
                                                       version,
                                                       soname,
                                                       nlohmann::json::parse(params),
                                                       cfg);
    }),
         py::arg("name"),
         py::arg("version"),
         py::arg("soname"),
         py::arg("params"),
         py::arg("cfg"))
    .def("payloadSize", &DrpAlgos::AlgorithmHost::payloadSize)
    .def("variableSize", &DrpAlgos::AlgorithmHost::variableSize)
    .def("process",
         [](DrpAlgos::AlgorithmHost& self, py::array_t<float> frame) {
           DrpAlgos::EventResult result;
           {
             py::gil_scoped_release release;
             result = self.process(frame.data());
           }

           return result;
         },
         py::arg("frame"))
    .def("process_many",
         [](DrpAlgos::AlgorithmHost& self, py::array_t<float> frames) {
           // Handling variably sized output will be harder I think...
           // Just deal with the fixed-size reducers here for now
           if (self.variableSize()) {
             throw std::runtime_error("Variably sized reducers do not support batch "
                                      "processing right now. Process each frame with "
                                      "the `process` method.");
           }

           const auto payload_size { static_cast<py::ssize_t>(self.payloadSize()) };
           const py::ssize_t num_frames { frames.shape(0) };
           const auto num_elem {
             static_cast<unsigned long long>(frames.size() / num_frames)
           };

           py::array_t<std::uint8_t> reduced_out({ num_frames, payload_size });
           std::uint8_t* dest_ptr { reduced_out.mutable_data() };
           const float* src_ptr { frames.data() };

           double deviceMs { 0.0 };
           {
             py::gil_scoped_release release;
             for (py::ssize_t i = 0; i < num_frames; ++i) {
               const auto res { self.process(src_ptr + i * num_elem) };

               std::memcpy(dest_ptr + i * payload_size,
                           res.payload.data(),
                           res.payload.size());
               deviceMs += res.deviceMs;
             }
           }

           // Print (device) processing time as well
           std::printf("mean device time per event: %.2f us\n",
                       1000.0 * deviceMs / num_frames);

           return reduced_out;
         },
         py::arg("frames"))
    .def("benchmark",
         &DrpAlgos::AlgorithmHost::benchmark,
         py::arg("iterations"));
}
