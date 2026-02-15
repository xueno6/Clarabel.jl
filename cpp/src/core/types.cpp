#include "clarabel/core/types.hpp"

namespace clarabel {
namespace core {

ValidationReport validate_problem_data(const ProblemData& data) {
    ValidationReport report;

    if (data.P.rows != data.P.cols) {
        report.issues.push_back({"P", "P must be square."});
    }
    if (data.P.rows != data.q.size()) {
        report.issues.push_back({"q", "q length must match P dimension."});
    }
    if (data.A.cols != data.q.size()) {
        report.issues.push_back({"A", "A column count must match q length."});
    }
    if (data.A.rows != data.b.size()) {
        report.issues.push_back({"b", "b length must match A row count."});
    }
    if (data.P.values.size() != data.P.rows * data.P.cols) {
        report.issues.push_back({"P.values", "P storage size must be rows*cols."});
    }
    if (data.A.row_ptr.size() != data.A.rows + 1) {
        report.issues.push_back({"A.row_ptr", "row_ptr size must be rows+1."});
    } else {
        if (!data.A.row_ptr.empty() && data.A.row_ptr.front() != 0) {
            report.issues.push_back({"A.row_ptr", "row_ptr must start at 0."});
        }
        for (std::size_t i = 1; i < data.A.row_ptr.size(); ++i) {
            if (data.A.row_ptr[i] < data.A.row_ptr[i - 1]) {
                report.issues.push_back({"A.row_ptr", "row_ptr must be nondecreasing."});
                break;
            }
        }
        if (!data.A.row_ptr.empty() && data.A.row_ptr.back() != data.A.values.size()) {
            report.issues.push_back({"A.row_ptr", "row_ptr back() must equal number of A values."});
        }
    }
    if (data.A.col_ind.size() != data.A.values.size()) {
        report.issues.push_back({"A.col_ind", "col_ind size must match values size."});
    }
    for (std::size_t c : data.A.col_ind) {
        if (c >= data.A.cols) {
            report.issues.push_back({"A.col_ind", "column index out of bounds."});
            break;
        }
    }

    return report;
}

ValidationReport validate_variables(const ProblemData& data, const Variables& vars, std::size_t expected_s_dim) {
    ValidationReport report;

    if (vars.x.size() != data.q.size()) {
        report.issues.push_back({"x", "x length must match q length."});
    }
    if (vars.y.size() != data.b.size()) {
        report.issues.push_back({"y", "y length must match b length."});
    }
    const std::size_t s_dim = expected_s_dim == 0 ? data.q.size() : expected_s_dim;
    if (vars.s.size() != s_dim) {
        report.issues.push_back({"s", "s length must match expected cone/slack dimension."});
    }

    return report;
}

} // namespace core
} // namespace clarabel
