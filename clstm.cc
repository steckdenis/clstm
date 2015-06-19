#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <stdarg.h>

#ifndef MAXEXP
#define MAXEXP 30
#endif

namespace ocropus {
char exception_message[256];

void throwf(const char *format, ...) {
    va_list arglist;
    va_start(arglist, format);
    vsprintf(exception_message, format, arglist);
    va_end(arglist);
    THROW(exception_message);
}

Assoc::Assoc(const string &s) {
    int start = 0;
    for (;; ) {
        int pos = s.find(":", start);
        string kvp;
        if (pos == string::npos) {
            kvp = s.substr(start);
            start = s.size();
        } else {
            kvp = s.substr(start, pos-start);
            start = pos+1;
        }
        int q = kvp.find("=");
        if (q == string::npos) THROW("no '=' in Assoc");
        string key = kvp.substr(0, q);
        string value = kvp.substr(q+1);
        (*this)[key] = value;
        if (start >= s.size()) break;
    }
}

map<string, ILayerFactory> layer_factories;

Network make_layer(const string &kind) {
    Network net;
    auto it = layer_factories.find(kind);
    if (it != layer_factories.end())
        net.reset(it->second());
    return net;
}

Network layer(const string &kind,
              int ninput,
              int noutput,
              const Assoc &args,
              const Networks &subs) {
    Network net;
    auto it = layer_factories.find(kind);
    if (it != layer_factories.end())
        net.reset(it->second());
    for (auto it : args) {
        net->attributes[it.first] = it.second;
    }
    net->attributes["ninput"] = std::to_string(ninput);
    net->attributes["noutput"] = std::to_string(noutput);
    for (int i = 0; i < subs.size(); i++)
        net->sub.push_back(subs[i]);
    net->initialize();
    return net;
}

template <class T>
int register_layer(const char *name) {
    T *net = new T();
    string kind = net->kind();
    delete net;
    string s(name);
    layer_factories[s] = [] () { return new T(); };
    layer_factories[kind] = [] () { return new T(); };
    return 0;
}
#define C(X, Y) X ## Y
#define REGISTER(X) int C(status_, X) = register_layer<X>(# X);

Mat debugmat;

using namespace std;
using Eigen::Ref;

bool no_update = false;
bool verbose = false;

void set_inputs(INetwork *net, Sequence &inputs) {
    net->inputs.like(inputs);
    for (int t = 0; t < net->inputs.size(); t++)
        net->inputs[t] = inputs[t];
}
void set_targets(INetwork *net, Sequence &targets) {
    int N = net->outputs.size();
    net->outputs.checkLike(targets);
    for (int t = 0; t < N; t++)
        net->outputs.d[t] = targets[t] - net->outputs[t];
}
void set_classes(INetwork *net, Classes &classes) {
    int N = net->outputs.size();
    assert(N == classes.size());
    net->outputs.d.resize(N);
    for (int t = 0; t < N; t++) {
        net->outputs.d[t] = -net->outputs[t];
        net->outputs.d[t](classes[t]) += 1;
    }
}
void train(INetwork *net, Sequence &xs, Sequence &targets) {
    assert(xs.size() > 0);
    assert(xs.size() == targets.size());
    net->inputs.copyValues(xs);
    net->forward();
    set_targets(net, targets);
    net->backward();
    net->update();
}
void ctrain(INetwork *net, Sequence &xs, Classes &cs) {
    net->inputs.copyValues(xs);
    net->forward();
    int len = net->outputs.size();
    assert(len > 0);
    int dim = net->outputs[0].size();
    assert(dim > 0);
    net->outputs.d.resize(len);
    if (dim == 1) {
        for (int t = 0; t < len; t++)
            net->outputs.d[t](0) = cs[t] ?
                1.0-net->outputs[t](0) : -net->outputs[t](0);
    } else {
        for (int t = 0; t < len; t++) {
            net->outputs.d[t] = -net->outputs[t];
            int c = cs[t];
            net->outputs.d[t](c) = 1-net->outputs[t](c);
        }
    }
    net->backward();
    net->update();
}
void cpred(INetwork *net, Classes &preds, Sequence &xs) {
    int N = xs.size();
    assert(COLS(xs[0]) == 0);
    net->inputs.copyValues(xs);
    preds.resize(N);
    net->forward();
    assert(net->outputs.size() == N);
    for (int t = 0; t < N; t++) {
        int index = -1;
        net->outputs[t].col(0).maxCoeff(&index);
        preds[t] = index;
    }
}

void INetwork::makeEncoders() {
    encoder.reset(new map<int, int>());
    for (int i = 0; i < codec.size(); i++) {
        encoder->insert(make_pair(codec[i], i));
    }
    iencoder.reset(new map<int, int>());
    for (int i = 0; i < icodec.size(); i++) {
        iencoder->insert(make_pair(icodec[i], i));
    }
}

void INetwork::encode(Classes &classes, const std::wstring &s) {
    if (!encoder) makeEncoders();
    classes.clear();
    for (int pos = 0; pos < s.size(); pos++) {
        unsigned c = s[pos];
        assert(encoder->count(c) > 0);
        c = (*encoder)[c];
        assert(c != 0);
        classes.push_back(c);
    }
}
void INetwork::iencode(Classes &classes, const std::wstring &s) {
    if (!iencoder) makeEncoders();
    classes.clear();
    for (int pos = 0; pos < s.size(); pos++) {
        int c = (*iencoder)[int(s[pos])];
        classes.push_back(c);
    }
}
std::wstring INetwork::decode(Classes &classes) {
    std::wstring s;
    for (int i = 0; i < classes.size(); i++)
        s.push_back(wchar_t(codec[classes[i]]));
    return s;
}
std::wstring INetwork::idecode(Classes &classes) {
    std::wstring s;
    for (int i = 0; i < classes.size(); i++)
        s.push_back(wchar_t(icodec[classes[i]]));
    return s;
}

void INetwork::info(string prefix){
    string nprefix = prefix + "." + name;
    cout << nprefix << ": " << learning_rate << " " << momentum << " ";
    cout << "in " << inputs.size() << " " << ninput() << " ";
    cout << "out " << outputs.size() << " " << noutput() << endl;
    for (auto s : sub) s->info(nprefix);
}

void INetwork::weights(const string &prefix, WeightFun f) {
    string nprefix = prefix + "." + name;
    myweights(nprefix, f);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->weights(nprefix+"."+to_string(i), f);
    }
}

void INetwork::states(const string &prefix, StateFun f) {
    string nprefix = prefix + "." + name;
    f(nprefix+".inputs", &inputs);
    f(nprefix+".outputs", &outputs);
    mystates(nprefix, f);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->states(nprefix+"."+to_string(i), f);
    }
}

void INetwork::networks(const string &prefix, function<void (string, INetwork*)> f) {
    string nprefix = prefix+"."+name;
    f(nprefix, this);
    for (int i = 0; i < sub.size(); i++) {
        sub[i]->networks(nprefix, f);
    }
}

Sequence *INetwork::getState(string name) {
    Sequence *result = nullptr;
    states("", [&result, &name](const string &prefix, Sequence *s) {
               if (prefix == name) result = s;
           });
    return result;
}

struct NetworkBase : INetwork {
    virtual void checkInputs() {
        assert(inputs.size() >= 1);
        assert(inputs.batchsize() >= 1);
        assert(inputs.nfeat() == ninput());
    }
    virtual void checkAll() {
        checkInputs();
        assert(outputs.size() >= 1);
        assert(outputs.batchsize() >= 1);
        assert(outputs.nfeat() == noutput());
        assert(outputs.size() == inputs.size());
        assert(outputs.batchsize() == inputs.batchsize());
    }
    Float error2(Sequence &xs, Sequence &targets) {
        inputs.copyValues(xs);
        forward();
        Float total = 0.0;
        outputs.checkLike(targets);
        for (int t = 0; t < outputs.size(); t++) {
            Vec delta = targets[t] - outputs[t];
            total += delta.array().square().sum();
            outputs.d[t] = delta;
        }
        backward();
        update();
        return total;
    }
};

inline Float limexp(Float x) {
#if 1
    if (x < -MAXEXP) return exp(-MAXEXP);
    if (x > MAXEXP) return exp(MAXEXP);
    return exp(x);
#else
    return exp(x);
#endif
}

inline Float sigmoid(Float x) {
#if 1
    return 1.0 / (1.0 + limexp(-x));
#else
    return 1.0 / (1.0 + exp(-x));
#endif
}

template <class NONLIN>
struct Full : NetworkBase {
    Mat W, d_W;
    Vec w, d_w;
    int nseq = 0;
    int nsteps = 0;
    string mykind = string("full_") + NONLIN::kind;
    Full() {
        name = mykind;
    }
    const char *kind() {
        return mykind.c_str();
    }
    int noutput() {
        return ROWS(W);
    }
    int ninput() {
        return COLS(W);
    }
    void initialize() {
        int no = irequire("noutput");
        int ni = irequire("ninput");
        randinit(W, no, ni, 0.01);
        randinit(w, no, 0.01);
        zeroinit(d_W, no, ni);
        zeroinit(d_w, no);
    }
    void forward() {
        checkInputs();
        int N = inputs.size();
        int bs = inputs.batchsize();
        outputs.like(inputs, noutput());
        for (int t = 0; t < inputs.size(); t++) {
            outputs[t] = MATMUL(W, inputs[t]);
            ADDCOLS(outputs[t], w);
            NONLIN::f(outputs[t]);
        }
        checkAll();
    }
    void backward() {
        checkAll();
        for (int t = outputs.d.size()-1; t >= 0; t--) {
            NONLIN::df(outputs.d[t], outputs[t]);
            inputs.d[t] = MATMUL_TR(W, outputs.d[t]);
        }
        int bs = COLS(inputs[0]);
        for (int t = 0; t < outputs.d.size(); t++) {
            d_W += MATMUL_RT(outputs.d[t], inputs[t]);
            for (int b = 0; b < bs; b++) d_w += COL(outputs.d[t], b);
        }
        nseq += 1;
        nsteps += outputs.d.size();
        outputs.d[0](0, 0) = NAN;  // invalidate it, since we have changed it
        checkAll();
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else THROW("unknown normalization");
        W += lr * d_W;
        w += lr * d_w;
        nsteps = 0;
        nseq = 0;
        d_W *= momentum;
        d_w *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, (Mat*)0);
        f(prefix+".w", &w, (Vec*)0);
    }
};

struct NoNonlin {
    static constexpr const char *kind = "linear";
    template <class T>
    static void f(T &x) {
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
    }
};
typedef Full<NoNonlin> LinearLayer;
REGISTER(LinearLayer);

struct SigmoidNonlin {
    static constexpr const char *kind = "sigmoid";
    template <class T>
    static void f(T &x) {
        x = MAPFUN(x, sigmoid);
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= y.array() * (1-y.array());
    }
};
typedef Full<SigmoidNonlin> SigmoidLayer;
REGISTER(SigmoidLayer);

Float tanh_(Float x) {
    return tanh(x);
}
struct TanhNonlin {
    static constexpr const char *kind = "tanh";
    template <class T>
    static void f(T &x) {
        x = MAPFUN(x, tanh_);
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= (1 - y.array().square());
    }
};
typedef Full<TanhNonlin> TanhLayer;
REGISTER(TanhLayer);

inline Float relu_(Float x) {
    return x <= 0 ? 0 : x;
}
inline Float heavi_(Float x) {
    return x <= 0 ? 0 : 1;
}
struct ReluNonlin {
    static constexpr const char *kind = "relu";
    template <class T>
    static void f(T &x) {
        x = MAPFUN(x, relu_);
    }
    template <class T, class U>
    static void df(T &dx, U &y) {
        dx.array() *= MAPFUN(y, heavi_).array();
    }
};
typedef Full<ReluNonlin> ReluLayer;
REGISTER(ReluLayer);

struct SoftmaxLayer : NetworkBase {
    Mat W, d_W;
    Vec w, d_w;
    int nsteps = 0;
    int nseq = 0;
    SoftmaxLayer() {
        name = "softmax";
    }
    const char *kind() {
        return "SoftmaxLayer";
    }
    int noutput() {
        return ROWS(W);
    }
    int ninput() {
        return COLS(W);
    }
    void initialize() {
        int no = irequire("noutput");
        int ni = irequire("ninput");
        if (no < 2) THROW("Softmax requires no>=2");
        randinit(W, no, ni, 0.01);
        randinit(w, no, 0.01);
        clearUpdates();
    }
    void clearUpdates() {
        int no = ROWS(W);
        int ni = COLS(W);
        zeroinit(d_W, no, ni);
        zeroinit(d_w, no);
    }
    void postLoad() {
        clearUpdates();
        makeEncoders();
    }
    void forward() {
        checkInputs();
        int N = inputs.size();
        outputs.like(inputs, noutput());
        for (int t = 0; t < N; t++) {
            for (int b = 0; b < COLS(outputs[t]); b++) {
                COL(outputs[t], b) = MAPFUN(DOT(W, COL(inputs[t], b)) + w, limexp);
                Float total = fmax(SUMREDUCE(COL(outputs[t], b)), 1e-9);
                COL(outputs[t], b) /= total;
            }
        }
        checkAll();
    }
    void backward() {
        checkAll();
        for (int t = outputs.d.size()-1; t >= 0; t--) {
            inputs.d[t] = MATMUL_TR(W, outputs.d[t]);
        }
        int bs = COLS(inputs[0]);
        for (int t = 0; t < outputs.d.size(); t++) {
            d_W += MATMUL_RT(outputs.d[t], inputs[t]);
            for (int b = 0; b < bs; b++) d_w += COL(outputs.d[t], b);
        }
        nsteps += outputs.d.size();
        nseq += 1;
        checkAll();
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else THROW("unknown normalization");
        W += lr * d_W;
        w += lr * d_w;
        nsteps = 0;
        nseq = 0;
        d_W *= momentum;
        d_w *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".W", &W, &d_W);
        f(prefix+".w", &w, &d_w);
    }
};
REGISTER(SoftmaxLayer);

struct Stacked : NetworkBase {
    Stacked() {
        name = "stacked";
    }
    const char *kind() {
        return "Stacked";
    }
    int noutput() {
        return sub[sub.size()-1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        checkInputs();
        assert(inputs.size() > 0);
        assert(sub.size() > 0);
        for (int n = 0; n < sub.size(); n++) {
            if (n == 0) sub[n]->inputs.copyValues(inputs);
            else sub[n]->inputs.copyValues(sub[n-1]->outputs);
            sub[n]->forward();
        }
        outputs.copyValues(sub[sub.size()-1]->outputs);
        assert(outputs.size() == inputs.size());
        checkAll();
    }
    void backward() {
        checkAll();
        for (int n = sub.size()-1; n >= 0; n--) {
            if (n+1 == sub.size()) sub[n]->outputs.d = outputs.d;
            else sub[n]->outputs.d = sub[n+1]->inputs.d;
            sub[n]->backward();
        }
        inputs.d = sub[0]->inputs.d;
        checkAll();
    }
    void update() {
        for (int i = 0; i < sub.size(); i++)
            sub[i]->update();
    }
};
REGISTER(Stacked);

template <class T>
inline void revcopy(vector<T> &out, vector<T> &in) {
    int N = in.size();
    //out.resize(N);
    assert(out.size() == in.size());
    for (int i = 0; i < N; i++) out[i] = in[N-i-1];
}

struct Reversed : NetworkBase {
    Reversed() {
        name = "reversed";
    }
    const char *kind() {
        return "Reversed";
    }
    int noutput() {
        return sub[0]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        checkInputs();
        assert(sub.size() == 1);
        INetwork *net = sub[0].get();
        net->inputs.like(inputs);
        revcopy(net->inputs.v, inputs.v);
        net->forward();
        outputs.like(net->outputs);
        revcopy(outputs.v, net->outputs.v);
        checkAll();
    }
    void backward() {
        checkAll();
        assert(sub.size() == 1);
        INetwork *net = sub[0].get();
        revcopy(net->outputs.d, outputs.d);
        net->backward();
        revcopy(inputs.d, net->inputs.d);
        checkAll();
    }
    void update() {
        sub[0]->update();
    }
};
REGISTER(Reversed);

struct Parallel : NetworkBase {
    Parallel() {
        name = "parallel";
    }
    const char *kind() {
        return "Parallel";
    }
    int noutput() {
        return sub[0]->noutput() + sub[1]->noutput();
    }
    int ninput() {
        return sub[0]->ninput();
    }
    void forward() {
        checkInputs();
        assert(sub.size() == 2);
        INetwork *net1 = sub[0].get();
        INetwork *net2 = sub[1].get();
        net1->inputs.copyValues(inputs);
        net2->inputs.copyValues(inputs);
        net1->forward();
        net2->forward();
        int n1 = net1->noutput();
        int n2 = net2->noutput();
        outputs.like(inputs, n1+n2);
        int N = outputs.size();
        int bs = net1->outputs.batchsize();
        net1->outputs.checkLike(outputs, net1->noutput());
        net2->outputs.checkLike(outputs, net2->noutput());
        for (int t = 0; t < N; t++) {
            assert(t < outputs.size());
            assert(outputs[t].rows() == n1+n2);
            assert(outputs[t].cols() == bs);
            BLOCK(outputs[t], 0, 0, n1, bs) = net1->outputs[t];
            BLOCK(outputs[t], n1, 0, n2, bs) = net2->outputs[t];
        }
        checkAll();
    }
    void backward() {
        checkAll();
        assert(sub.size() == 2);
        INetwork *net1 = sub[0].get();
        INetwork *net2 = sub[1].get();
        net1->outputs.checkLike(outputs, net1->noutput());
        net2->outputs.checkLike(outputs, net2->noutput());
        int n1 = net1->outputs.nfeat();
        int n2 = net2->outputs.nfeat();
        int bs = net1->outputs.batchsize();
        int N = outputs.size();
        for (int t = 0; t < N; t++) {
            net1->outputs.d[t] = BLOCK(outputs.d[t], 0, 0, n1, bs);
            net2->outputs.d[t] = BLOCK(outputs.d[t], n1, 0, n2, bs);
        }
        net1->backward();
        net2->backward();
        inputs.checkLike(outputs, ninput());
        for (int t = 0; t < N; t++) {
            inputs.d[t] = net1->inputs.d[t];
            inputs.d[t] += net2->inputs.d[t];
        }
        checkAll();
    }
    void update() {
        for (int i = 0; i < sub.size(); i++) sub[i]->update();
    }
};
REGISTER(Parallel);

namespace {
template <class NONLIN, class T>
inline Mat nonlin(T &a) {
    Mat result = a;
    NONLIN::f(result);
    return result;
}
template <class NONLIN, class T>
inline Mat yprime(T &a) {
    Mat result = Mat::Ones(ROWS(a), COLS(a));
    NONLIN::df(result, a);
    return result;
}
template <class NONLIN, class T>
inline Mat xprime(T &a) {
    Mat result = Mat::Ones(ROWS(a), COLS(a));
    Mat temp = a;
    NONLIN::f(temp);
    NONLIN::df(result, temp);
    return result;
}
template <typename F, typename T>
void each(F f, T &a) {
    f(a);
}
template <typename F, typename T, typename ... Args>
void each(F f, T &a, Args&&... args) {
    f(a);
    each(f, args ...);
}
}

#if 0
void f_delay(Mat &out, Sequence &in, int t) {
    if (t > 0) {
        out = in[t-1];
        return;
    } else {
        out.setZero(in[0].rows(), in[0].cols());
    }
}
void b_delay(StepRef out, Sequence &in, int t) {
    if (t < in.size()-1) {
        in(t).d += out.d;
    }
}
void f_1stack(Mat &out, Mat &a, Mat &b) {
    int ni = ROWS(a);
    int no = ROWS(b);
    int nf = ni+no+1;
    int bs = COLS(a);
    BLOCK(out, 0, 0, 1, bs).setConstant(1);
    BLOCK(out, 1, 0, ni, bs) = a;
    BLOCK(source[t], 1+ni, 0, no, bs) = b;
}
void b_1stack(Mat &out, Mat &a, Mat &b) {
    int ni = ROWS(a);
    int no = ROWS(b);
    int nf = ni+no+1;
    int bs = COLS(a);
    a.d += BLOCK(out.d, 1, 0, ni, bs);
    b.d += BLOCK(out.d, 1+ni, 0, no, bs);
}
template <class F>
void f_hlayer(Mat &out, Mat &in, Mat &w) {
    out = nonlin<F>(MATMUL(W, source));
}
template <class F>
void b_hlayer(StepRef out, StepRef in, Mat &W, Mat &DW) {
    Mat deltas;
    deltas = EMUL(yprime<F>(out.v), out.d);
    in.d += MATMUL_TR(W, deltas);
    DW += MATMUL_RT(in.v, deltas);
}
void f_mul(Mat &out, Mat &a, Mat &b) {
    out = EMUL(a, b);
}
void b_mul(StepRef &out, StepRef &a, StepRef &b) {
    a.d += EMUL(out.d, b.v);
    b.d += EMUL(out.d, a.v);
}
void f_add(Mat &out, Mat &a, Mat &b) {
    out = a + b;
}
void b_add(StepRef &out, StepRef &a, StepRef &b) {
    a.d += out.d;
    b.d += out.d;
}
template <class F>
void f_nonlin(Mat &out, Mat &in) {
    out = nonlin<F>(in);
}
template <class F>
void b_nonlin(StepRef &out, StepRef &in) {
    in.d += yprime<F>(out.d);
}
#endif

template <class F = SigmoidNonlin, class G = TanhNonlin, class H = TanhNonlin>
struct GenericNPLSTM : NetworkBase {
#define WEIGHTS WGI, WGF, WGO, WCI
#define DWEIGHTS DWGI, DWGF, DWGO, DWCI
    Sequence source, gi, gf, go, ci, state;
    Mat WEIGHTS, DWEIGHTS;
    Float gradient_clipping = 10.0;
    int ni, no, nf;
    int nsteps = 0;
    int nseq = 0;
    string mykind = string("NPLSTM_")+F::kind+G::kind+H::kind;
    GenericNPLSTM() {
        name = "lstm";
    }
    const char *kind() {
        return mykind.c_str();
    }
    int noutput() {
        return no;
    }
    int ninput() {
        return ni;
    }
    void postLoad() {
        no = ROWS(WGI);
        nf = COLS(WGI);
        assert(nf > no);
        ni = nf-no-1;
        clearUpdates();
    }
    void initialize() {
        int ni = irequire("ninput");
        int no = irequire("noutput");
        int nf = 1+ni+no;
        string mode = attr("weight_mode", "pos");
        float weight_dev = dattr("weight_dev", 0.01);
        this->ni = ni;
        this->no = no;
        this->nf = nf;
        each([weight_dev, mode, no, nf](Mat &w) {
                 randinit(w, no, nf, weight_dev, mode);
             }, WEIGHTS);
#if 0
        float gf_mean = dattr("gf_mean", 0.0);
        float gf_dev = dattr("gf_dev", 0.01);
        Vec offset;
        randinit(offset, no, gf_dev, mode);
        offset.array() += gf_mean;
        COL(WGF, 0) = offset;
#endif
        clearUpdates();
    }
    void clearUpdates() {
        each([this](Mat &d) { d = Mat::Zero(no, nf); }, DWEIGHTS);
    }
    void resize(int N, int bs) {
        assert(N >= 1);
        assert(bs >= 1);
        source.resize(N, nf, bs);
        state.resize(N, no, bs);
        gi.resize(N, no, bs);
        gf.resize(N, no, bs);
        go.resize(N, no, bs);
        ci.resize(N, no, bs);
        outputs.resize(N, no, bs);
    }
#define A array()
    void forward() {
        checkInputs();
        int N = inputs.size();
        int bs = inputs.batchsize();
        resize(N, bs);
        for (int t = 0; t < N; t++) {
#if 1
            int bs = COLS(inputs[t]);
            BLOCK(source[t], 0, 0, 1, bs).setConstant(1);
            BLOCK(source[t], 1, 0, ni, bs) = inputs[t];
            if (t == 0) BLOCK(source[t], 1+ni, 0, no, bs).setConstant(0);
            else BLOCK(source[t], 1+ni, 0, no, bs) = outputs[t-1];
            gi[t] = nonlin<F>(MATMUL(WGI, source[t]));
            gf[t] = nonlin<F>(MATMUL(WGF, source[t]));
            go[t] = nonlin<F>(MATMUL(WGO, source[t]));
            ci[t] = nonlin<G>(MATMUL(WCI, source[t]));
            state[t] = ci[t].A * gi[t].A;
            if (t > 0) state[t] += EMUL(gf[t], state[t-1]);
            outputs[t] = nonlin<H>(state[t]).A * go[t].A;
#else

            Mat last_out, last_state;
            Mat from_input, from_state;
            Mat linout;
            f_delay(last_out, outputs, t);
            f_1stack(source[t], inputs[t], last_out);
            f_hlayer<F>(gi[t], source[t], WGI);
            f_hlayer<F>(gf[t], source[t], WGF);
            f_hlayer<F>(go[t], source[t], WGO);
            f_hlayer<G>(ci[t], source[t], WCI);
            f_delay(last_state, state, t);
            f_mul(from_input, gi[t], ci[t]);
            f_mul(from_state, gf[t], last_state);
            f_add(state[t], from_input, from_state);
            f_mul<H>(linout, go[t], state[t]);
            f_nolin<H>(output[t], linout);
#endif
            checkAll();
        }
    }
    void backward() {
        checkAll();
        int N = inputs.size();
        for (int t = N-1; t >= 0; t--) {
#if 1
            int bs = COLS(outputs.d[t]);
            if (t < N-1) outputs.d[t] += BLOCK(source.d[t+1], 1+ni, 0, no, bs);
            go.d[t] = EMUL(EMUL(yprime<F>(go[t]), nonlin<H>(state[t])), outputs.d[t]);
            state.d[t] = EMUL(EMUL(xprime<H>(state[t]), go[t].A), outputs.d[t]);
            if (t < N-1) state.d[t] += EMUL(state.d[t+1], gf[t+1]);
            if (t > 0) gf.d[t] = EMUL(EMUL(yprime<F>(gf[t]), state.d[t]), state[t-1]);
            gi.d[t] = EMUL(EMUL(yprime<F>(gi[t]), state.d[t]), ci[t]);
            ci.d[t] = EMUL(EMUL(yprime<G>(ci[t]), state.d[t]), gi[t]);
            source.d[t] = MATMUL_TR(WGI, gi.d[t]);
            if (t > 0) source.d[t] += MATMUL_TR(WGF, gf.d[t]);
            source.d[t] += MATMUL_TR(WGO, go.d[t]);
            source.d[t] += MATMUL_TR(WCI, ci.d[t]);
            inputs.d[t] = BLOCK(source.d[t], 1, 0, ni, bs);
#else
            Mat last_out, last_state;
            Mat from_input, from_state;
            Mat linout;
            f_nolin<H>(output[t], linout);
            f_mul<H>(linout, go[t], state[t]);
            f_add(state[t], from_input, from_state);
            f_mul(from_state, gf[t], last_state);
            f_mul(from_input, gi[t], ci[t]);
            f_delay(last_state, state, t);
            f_hlayer<G>(ci[t], source[t], WCI);
            f_hlayer<F>(go[t], source[t], WGO);
            f_hlayer<F>(gf[t], source[t], WGF);
            f_hlayer<F>(gi[t], source[t], WGI);
            f_1stack(source[t], inputs[t], last_out);
            f_delay(last_out, outputs, t);
#endif
        }
        if (gradient_clipping > 0 || gradient_clipping < 999) {
            gradient_clip(gi.d, gradient_clipping);
            gradient_clip(gf.d, gradient_clipping);
            gradient_clip(go.d, gradient_clipping);
            gradient_clip(ci.d, gradient_clipping);
        }
        for (int t = 0; t < N; t++) {
            DWGI += MATMUL_RT(gi.d[t], source[t]);
            if (t > 0) DWGF += MATMUL_RT(gf.d[t], source[t]);
            DWGO += MATMUL_RT(go.d[t], source[t]);
            DWCI += MATMUL_RT(ci.d[t], source[t]);
        }
        nsteps += N;
        nseq += 1;
        checkAll();
    }
#undef A
    void gradient_clip(vector<Mat> &s, Float m=1.0) {
        for (int t = 0; t < s.size(); t++) {
            s[t] = MAPFUNC(s[t],
                           [m](Float x) {
                               return x > m ? m : x < -m ? -m : x;
                           });
        }
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else THROW("unknown normalization");
        WGI += lr * DWGI;
        WGF += lr * DWGF;
        WGO += lr * DWGO;
        WCI += lr * DWCI;
        DWGI *= momentum;
        DWGF *= momentum;
        DWGO *= momentum;
        DWCI *= momentum;
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".WGI", &WGI, &DWGI);
        f(prefix+".WGF", &WGF, &DWGF);
        f(prefix+".WGO", &WGO, &DWGO);
        f(prefix+".WCI", &WCI, &DWCI);
    }
    virtual void mystates(const string &prefix, StateFun f) {
        f(prefix+".inputs", &inputs);
        f(prefix+".outputs", &outputs);
        f(prefix+".state", &state);
        f(prefix+".gi", &gi);
        f(prefix+".go", &go);
        f(prefix+".gf", &gf);
        f(prefix+".ci", &ci);
    }
    Sequence *getState() {
        return &state;
    }
};
typedef GenericNPLSTM<> NPLSTM;
REGISTER(NPLSTM);

INetwork *make_SigmoidLayer() {
    return new SigmoidLayer();
}
INetwork *make_SoftmaxLayer() {
    return new SoftmaxLayer();
}
INetwork *make_ReluLayer() {
    return new ReluLayer();
}
INetwork *make_Stacked() {
    return new Stacked();
}
INetwork *make_Reversed() {
    return new Reversed();
}
INetwork *make_Parallel() {
    return new Parallel();
}
INetwork *make_LSTM() {
    return new NPLSTM();
}
INetwork *make_NPLSTM() {
    return new NPLSTM();
}

inline Float log_add(Float x, Float y) {
    if (abs(x-y) > 10) return fmax(x, y);
    return log(exp(x-y)+1) + y;
}

inline Float log_mul(Float x, Float y) {
    return x+y;
}

void forward_algorithm(Mat &lr, Mat &lmatch, double skip) {
    int n = ROWS(lmatch), m = COLS(lmatch);
    lr.resize(n, m);
    Vec v(m), w(m);
    for (int j = 0; j < m; j++) v(j) = skip * j;
    for (int i = 0; i < n; i++) {
        w.segment(1, m-1) = v.segment(0, m-1);
        w(0) = skip * i;
        for (int j = 0; j < m; j++) {
            Float same = log_mul(v(j), lmatch(i, j));
            Float next = log_mul(w(j), lmatch(i, j));
            v(j) = log_add(same, next);
        }
        lr.row(i) = v;
    }
}

void forwardbackward(Mat &both, Mat &lmatch) {
    Mat lr;
    forward_algorithm(lr, lmatch);
    Mat rlmatch = lmatch;
    rlmatch = rlmatch.rowwise().reverse().eval();
    rlmatch = rlmatch.colwise().reverse().eval();
    Mat rl;
    forward_algorithm(rl, rlmatch);
    rl = rl.colwise().reverse().eval();
    rl = rl.rowwise().reverse().eval();
    both = lr + rl;
}

void ctc_align_targets(Mat &posteriors, Mat &outputs, Mat &targets) {
    double lo = 1e-5;
    int n1 = ROWS(outputs);
    int n2 = ROWS(targets);
    int nc = COLS(targets);

    // compute log probability of state matches
    Mat lmatch;
    lmatch.resize(n1, n2);
    for (int t1 = 0; t1 < n1; t1++) {
        Vec out = outputs.row(t1);
        out = out.cwiseMax(lo);
        out /= out.sum();
        for (int t2 = 0; t2 < n2; t2++) {
            double value = out.transpose() * targets.row(t2).transpose();
            lmatch(t1, t2) = log(value);
        }
    }
    // compute unnormalized forward backward algorithm
    Mat both;
    forwardbackward(both, lmatch);

    // compute normalized state probabilities
    Mat epath = (both.array() - both.maxCoeff()).unaryExpr(ptr_fun(limexp));
    for (int j = 0; j < n2; j++) {
        double l = epath.col(j).sum();
        epath.col(j) /= l == 0 ? 1e-9 : l;
    }
    debugmat = epath;

    // compute posterior probabilities for each class and normalize
    Mat aligned;
    aligned.resize(n1, nc);
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < nc; j++) {
            double total = 0.0;
            for (int k = 0; k < n2; k++) {
                double value = epath(i, k) * targets(k, j);
                total += value;
            }
            aligned(i, j) = total;
        }
    }
    for (int i = 0; i < n1; i++) {
        aligned.row(i) /= fmax(1e-9, aligned.row(i).sum());
    }

    posteriors = aligned;
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Sequence &targets) {
    assert(COLS(outputs[0]) == 1);
    assert(COLS(targets[0]) == 1);
    int n1 = outputs.size();
    int n2 = targets.size();
    int nc = targets[0].size();
    Mat moutputs(n1, nc);
    Mat mtargets(n2, nc);
    for (int i = 0; i < n1; i++) moutputs.row(i) = outputs[i].col(0);
    for (int i = 0; i < n2; i++) mtargets.row(i) = targets[i].col(0);
    Mat aligned;
    ctc_align_targets(aligned, moutputs, mtargets);
    posteriors.resize(aligned.rows(), aligned.cols(), 1);
    for (int i = 0; i < n1; i++) {
        posteriors[i].col(0) = aligned.row(i);
    }
}

void ctc_align_targets(Sequence &posteriors, Sequence &outputs, Classes &targets) {
    int nclasses = outputs[0].size();
    Sequence stargets;
    stargets.resize(targets.size(), nclasses, 1);
    for (int t = 0; t < stargets.size(); t++) {
        stargets[t](targets[t], 0) = 1.0;
    }
    ctc_align_targets(posteriors, outputs, stargets);
}

void mktargets(Sequence &seq, Classes &transcript, int ndim) {
    seq.resize(2*transcript.size()+1, ndim, 1);
    for (int t = 0; t < seq.size(); t++) {
        seq[t].setZero(ndim, 1);
        if (t%2 == 1) seq[t](transcript[(t-1)/2]) = 1;
        else seq[t](0) = 1;
    }
}

void trivial_decode(Classes &cs, Sequence &outputs, int batch) {
    int N = outputs.size();
    int t = 0;
    float mv = 0;
    int mc = -1;
    while (t < N) {
        int index;
        float v = outputs[t].col(batch).maxCoeff(&index);
        if (index == 0) {
            // NB: there should be a 0 at the end anyway
            if (mc != -1 && mc != 0) cs.push_back(mc);
            mv = 0; mc = -1; t++;
            continue;
        }
        if (v > mv) {
            mv = v;
            mc = index;
        }
        t++;
    }
}

void ctc_train(INetwork *net, Sequence &xs, Sequence &targets) {
    // untested
    assert(COLS(xs[0]) == 1);
    assert(xs.size() <= targets.size());
    assert(!anynan(xs));
    net->inputs.copyValues(xs);
    net->forward();
    if (anynan(net->outputs)) THROW("got NaN");
    Sequence aligned;
    ctc_align_targets(aligned, net->outputs, targets);
    if (anynan(aligned)) THROW("got NaN");
    set_targets(net, aligned);
    net->backward();
}

void ctc_train(INetwork *net, Sequence &xs, Classes &targets) {
    // untested
    Sequence ys;
    mktargets(ys, targets, net->noutput());
    ctc_train(net, xs, ys);
}

void ctc_train(INetwork *net, Sequence &xs, BatchClasses &targets) {
    THROW("unimplemented");
}
}  // namespace ocropus

#ifdef CLSTM_EXTRAS
// Extra layers; this uses internal function and class definitions from this
// file, so it's included rather than linked. It's mostly a way of slowly deprecating
// old layers.
#include "clstm_extras.i"
#endif
