from pycoin.tx.Tx import Tx 
from pycoin.tx.TxIn import TxIn, TxInGeneration
from pycoin.tx.TxOut import TxOut
from pycoin.tx.UnsignedTx import UnsignedTx, UnsignedTxOut
from pycoin.serialize import b2h_rev, b2h
from pycoin.block import Block
from pycoin.tx.script.tools import disassemble, compile
from pycoin.encoding import hash160_sec_to_bitcoin_address, h2b, public_pair_to_hash160_sec, sec_to_public_pair, is_sec_compressed
from pycoin.serialize.bitcoin_streamer import parse_struct
from pycoin.encoding import wif_to_secret_exponent
from pycoin.tx.script.solvers import SecretExponentSolver
from pycoin.ecdsa import public_pair_for_secret_exponent, generator_secp256k1

from jsonrpc.proxy import ServiceProxy
from decimal import Decimal
import binascii
from cStringIO import StringIO
from types import StringTypes
import socket
import logging
logger = logging.getLogger('pycoin.proxy')

def hexreverse (h):
    return "".join([h[x:x+2] for x in range(0,len(h),2)][::-1])

def floatToBtcInt (val):
    return int(float(val)*1e8)

class ChainDefs (object):
    def __init__ (self, chainid, addrversion, port):
        self.chainid = chainid
        self.addrversion = addrversion
        self.port = port

CHAINS = {'btc': ChainDefs('btc', b'\0', 8332),
          'ltc': ChainDefs('btc', b'\x30', 9332),
          'nmc': ChainDefs('btc', b'\x34', 8336),
          'trc': ChainDefs('btc', b'\0', 13332),
          }

MAXCONF=99999999

class NotEnoughFundsException (Exception):
    def __init__ (self, needed, have):
        Exception.__init__(self, "Not Enough Funds, needed %0.8f, have %0.8f" % (needed, have))
        self.needed = needed
        self.have = have

#class NotEnoughValueForFeeException (Exception):
#    def __init__ (self, value, fee):
#        Exception.__init__(self, "Invalid Amount, value=%s fee=%f" % (value, fee))
#        
#
#class NotEnoughFeesException (Exception):
#    pass
#
#class TransactionSigningException (Exception):
#    pass

class nkey (str):
  def __new__(self, content, hash):
    self = str.__new__(self, content)
    self.h = hash
    return self
  def __hash__ (self):
    return self.h

class ChainProxy (object):
    def __init__ (self, chainid, host='localhost', port=None, user=None, passwd=None):
        chaindefs = CHAINS[chainid]
        self.chainid = chainid
        self.addrversion = chaindefs.addrversion
        self.host = host
        if port:
            self.port = port
        else:
            self.port = chaindefs.port
        self.user = user
        self.passwd = passwd
        self._conn = None

    def getConn (self):
        if self._conn is None:
            auth = ''
            if self.user is not None:
                auth = self.user
            if self.passwd is not None:
                auth += ':%s' % self.passwd
            if len(auth) > 0:
                auth += '@'
            self._conn = ServiceProxy("http://%s%s:%d" % (auth, self.host, self.port))
            print "http://%s%s:%d" % (auth, self.host, self.port)
        return self._conn

    def getBlock (self, hash_or_height, shallow_txins=False):
        return ProxyBlock.from_blockhash(self, hash_or_height, shallow_txins=shallow_txins)


    def getTx (self, txhash, shallow_txins=False):
        return ProxyTx.from_txhash(self, txhash, shallow_txins=shallow_txins)

    def getAddressBalance (self, address, minconf=0, maxconf=MAXCONF):
        conn = self.getConn()
        rv = conn.listunspent(minconf,maxconf,[address])
        return sum(map(lambda x: x['amount'], rv))

    def getNewAddress (self):
        
        return None
        
    def listUnspent(self, addresses, minconf=0):
        if isinstance(addresses, StringTypes):
            addresses = [addresses]
        conn = self.getConn()
        rv = conn.listunspent(minconf,MAXCONF,addresses)
        rv.sort(key=lambda x: x['confirmations'], reverse=True)
        return rv

    def chooseInputs (self, value, unspent):
        """ selects the least amount of inputs to cover value.
        Currently this means getting the smallest input that is larger than value and
        if one isn't available grab the largest input that is smaller and repeat.
        value is Decimal, unspent is a list of unspent tx as listUnspent returns
        """
        rv = [[], Decimal(0)]
        unspent.sort(key=lambda x: x['amount'])
        uslice = filter(lambda x: x['amount'] >= value, unspent)
        if len(uslice):
            rv[0].append(uslice[0])
            rv[1] += uslice[0]['amount']
        elif len(unspent):
            rv[0].append(unspent[-1])
            rv[1] += unspent[-1]['amount']
            trv = self.chooseInputs(value - rv[1], unspent[:-1])
            rv[0].extend(trv[0])
            rv[1] += trv[1]
        return rv

    def mkTx (self, coins_to, coins_from, addresses=[], changeaddress=None, fee=0, minconf=0, aggregate=True):
        """
        coins_to: [(intvalue, address),...] - what to pay
        coins_from: [(previous_hash, previous_index, coin_value),...] - outputs to use by default
        addresses: [address, ...] - own addresses that we can get more outputs from if needed
        changeaddress: the address to send change to, if any. If None, the first entry in addresses is used
        fee: the satoshi amount to pay as fees
        minconf: if extra txs are used from addresses, limit to those with at least minconf confirmations
        aggregate: if multiple payments to the same address are done, do we do one per address with the sum (aggregate) or
                   keep them separate?
        """
        # collect outputs
        outvalue = Decimal('0')
        outputs = {}
        for v,a in coins_to:
            v = Decimal(int(v)) / 10**8
            outvalue += v
            out = outputs.get(a, [])
            out.append(v)
            outputs[a] = out
        # collect inputs
        inputs = []
        invalue = Decimal('0')
        for h,n,v in coins_from:
            invalue += Decimal(int(v)) / 10**8
            inputs.append({'txid':h, 'vout':n})
        # add fee
        outvalue += Decimal(int(fee)) / 10**8     
        # if needed add more inputs to cover the output value
        if invalue < outvalue:
            try:
                # need more inputs
                newtx, newtxtotal = self.chooseInputs(outvalue-invalue, self.listUnspent(addresses, minconf=minconf))
            except socket.error:
                # error communicating with processor, so just abort this one for later retry
                return None
            invalue += newtxtotal
            inputs.extend([{'txid': txin['txid'], 'vout': txin['vout']} for txin in newtx])
        if invalue < outvalue:
            raise NotEnoughFundsException(outvalue, invalue)
        # handle change
        if invalue > outvalue:
            change = invalue - outvalue
            if changeaddress is None:
                if len(addresses) == 0:
                    raise ValueError ('Need a change address')
                changeaddress = addresses[0]
            outvalue += change
            out = outputs.get(changeaddress, [])
            out.append(change)
            outputs[changeaddress] = out
        # convert outputs dict to something json encodable
        noutputs = {}
        khash = 0
        for k,v in outputs.items():
            if aggregate:
                noutputs[nkey(k,khash)] = float(sum(v))
                khash += 1
            else:
                for e in v:
                    noutputs[nkey(k,khash)] = float(e)
                    khash += 1
        # finally, build the tx
        conn = self.getConn()
        raw = conn.createrawtransaction(inputs, noutputs)
        return ProxyUnsignedTx.from_hex(self, raw)

    def signTx (self, unsignedtx, keystore):
        skeys = [keystore.get(self.chainid,x) for x in unsignedtx.txs_in_addresses(self)]
        if None in skeys:
            raise ValueError ("Don't have keys for all the input addresses: %s" % (", ".join(unsignedtx.txs_in_addresses(self))))
        return unsignedtx.simple_sign(skeys)

class ProxyBlock (Block):
    @classmethod
    def from_blockhash (klass, proxy, blockhash, shallow_txins=False):
        conn = proxy.getConn()
        if isinstance(blockhash, (int, long)) or len(blockhash)<10 and blockhash.isdigit():
            # likely block height
            height = int(blockhash)
            blockhash = conn.getblockhash(height)
        else:
            height = 0
        blockjson = conn.getblock(blockhash)
        txs = []
        for e,txhash in enumerate(blockjson['tx']):
            txs.append(ProxyTx.from_txhash(proxy, txhash,
                                      is_first_in_block=(e==0),
                                      shallow_txins=shallow_txins
                                      ))
        rv = klass(version = int(blockjson['version']),
                   previous_block_hash = binascii.unhexlify(hexreverse(blockjson['previousblockhash'])),
                   merkle_root = binascii.unhexlify(hexreverse(blockjson['merkleroot'])),
                   timestamp = long(blockjson['time']),
                   difficulty = float(blockjson.get('difficulty',0)),
                   nonce = long(blockjson['nonce']),
                   txs = txs)
        rv._hash = binascii.unhexlify(hexreverse(blockhash))
        rv.height = blockjson.get('height', height)
        rv.reorg = False
        return rv

class ProxyTxIn (TxIn):
    is_shallow = True
    def address (self):
        return getattr(self, '_address', None)
    
    def query_links (self, proxy):
        if self.previous_hash != 0:
            tx = ProxyTx.from_txhash(proxy, b2h_rev(self.previous_hash), do_query_links=False)
            txout = tx.txs_out[self.previous_index]
            self._address = txout.address(addrversion=proxy.addrversion)
            self.coin_value = txout.coin_value
            self.is_shallow = False

class ProxyTxOut (TxOut):
    def address_h160 (self, addrversion=b'\0'):
        if not hasattr(self, '_address_h160'):
            self.address(addrversion)
        return self._address_h160

    @staticmethod
    def address_h160_from_script (script):
        s = disassemble(script).split(' ')
        if 'OP_HASH160' in s:
            p = s.index('OP_HASH160')
            if len(s) > p+1:
                return h2b(s[p+1])
        elif 'OP_CHECKSIG' in s:
            p = s.index('OP_CHECKSIG')
            if len(s[p-1]) in (66, 130):
                # public key
                sec = h2b(s[p-1])
                return public_pair_to_hash160_sec(sec_to_public_pair(sec), is_sec_compressed(sec))
        else:
            logger.warn("Can't parse address from script: %s" % (s))
            return None

    def address (self, addrversion=b'\0'):
        if not hasattr(self, '_address'):
            self._address_h160 = ProxyTxOut.address_h160_from_script(self.script)
            if self._address_h160 is None:
                self._address = None
            else:
                self._address = hash160_sec_to_bitcoin_address(self._address_h160, addrversion=addrversion)
        return self._address

class ProxyTx (Tx):
    @classmethod
    def from_txhash (klass, proxy, txhash, is_first_in_block=False, do_query_links=True, shallow_txins=False):
        conn = proxy.getConn()
        rawtx = conn.getrawtransaction(txhash)
        rawtx = binascii.unhexlify(rawtx)
        f = StringIO(rawtx)
        rv = klass.parse(f, is_first_in_block=is_first_in_block)
        rv.rawsize = len(rawtx)
        if do_query_links:
            rv.query_links(proxy, shallow_txins=shallow_txins)
        return rv

    @classmethod
    def parse(self, f, is_first_in_block=False):
        """Parse a Bitcoin transaction Tx from the file-like object f."""
        version, count = parse_struct("LI", f)
        txs_in = []
        if is_first_in_block:
            txs_in.append(TxInGeneration.parse(f))
            count = count - 1
        for i in range(count):
            txs_in.append(ProxyTxIn.parse(f))
        count, = parse_struct("I", f)
        txs_out = []
        for i in range(count):
            txs_out.append(ProxyTxOut.parse(f))
        lock_time, = parse_struct("L", f)
        rv = self(version, txs_in, txs_out, lock_time)
        for txe in rv.txs_in + rv.txs_out:
            txe.tx = rv
        return rv
    
    def fees (self, proxy, quick=False):
        if not hasattr(self, '_fees'):
            if quick:
                conn = proxy.getConn()
                rpctx = conn.gettransaction(self.id())
                self._fees = floatToBtcInt(rpctx.get('fee', 0))
            else:
                self.query_links(proxy)
                self._fees = self.amount_in - self.amount_out
        return self._fees

    def query_links (self, proxy, shallow_txins=False):
        self.amount_in = 0
        self.amount_out = 0
        for txin in self.txs_in:
            if getattr(txin, 'is_shallow', True) is False:
                # already deep
                self.amount_in += txin.coin_value
                continue
            if isinstance(txin, TxInGeneration) or shallow_txins:
                txin.coin_value = 0
                txin.is_shallow = True
                continue
            txin.query_links(proxy)
            self.amount_in += txin.coin_value
        for txout in self.txs_out:
            txout.address(addrversion=proxy.addrversion)
            self.amount_out += txout.coin_value

    def getTrustData (self, proxy):
        conn = proxy.getConn()
        if not hasattr(self, 'rawsize'):
            self.rawsize = len(conn.getrawtransaction(self.id())) / 2
        rpctx = conn.gettransaction(self.id())
        confirm_trust = rpctx.get('confirmations', 0)
        fees = floatToBtcInt(rpctx.get('fee', 0))
        return {'depth': confirm_trust, 'size': self.rawsize, 'fees': fees}

    def submit (self, proxy):
        tx = self.to_txhash()
        conn = proxy.getConn()
        #txhash = conn.sendrawtransaction(tx)

    def to_txhash (self):
        tx = StringIO()
        self.stream(tx)
        return b2h(tx.getvalue())

    def simple_validate (self, proxy):
        txcache = {}
        def get_txout (txhash, vout):
            if not txcache.has_key(txhash):
                tx = ProxyTx.from_txhash(proxy, b2h_rev(txhash), do_query_links=False, shallow_txins=True)
                txcache[txhash] = tx
            return txcache[txhash].txs_out[vout]
        return self.validate(get_txout)

class OfflineAwareSolver (SecretExponentSolver):
    def __init__(self, secret_exponents):
        super(OfflineAwareSolver, self).__init__(secret_exponents)
        STANDARD_SCRIPT_OUT = "OP_DUP OP_HASH160 %s OP_EQUALVERIFY OP_CHECKSIG"
        self.script_hash160 = []
        for se in secret_exponents:
            pp = public_pair_for_secret_exponent(generator_secp256k1, se)
            h160sec = public_pair_to_hash160_sec(pp, False)
            script_text = STANDARD_SCRIPT_OUT % b2h(h160sec)
            print script_text
            self.script_hash160.append(compile(script_text))

class ProxyUnsignedTx (UnsignedTx):
    @classmethod
    def from_hex (klass, proxy, hexv):
        STANDARD_SCRIPT_OUT = "OP_DUP OP_HASH160 %s OP_EQUALVERIFY OP_CHECKSIG"
        tx = ProxyTx.parse(StringIO(binascii.unhexlify(hexv)))
        uto = []
        for txi in tx.txs_in:
            if proxy is None:
                pscript = ''
            else:
                thistx = proxy.getTx(b2h_rev(txi.previous_hash), shallow_txins=True)
                pscript = thistx.txs_out[txi.previous_index].script
            uto.append(UnsignedTxOut(txi.previous_hash, txi.previous_index, None, pscript))

        uti = []
        for txo in tx.txs_out:
            script_text = STANDARD_SCRIPT_OUT % b2h(txo.address_h160())
            script_bin = compile(script_text)
            uti.append(ProxyTxOut(txo.coin_value, script_bin))

        return klass(tx.version, uto, uti, tx.lock_time)

    def simple_sign (self, skeys):
        solver = OfflineAwareSolver([wif_to_secret_exponent(x) for x in skeys])
        for idx, tx in enumerate(self.unsigned_txs_out):
            if tx.script == '':
                tx.script = solver.script_hash160[idx]
        return self.sign(solver, TxClass=ProxyTx)

    def txs_in_addresses (self, proxy):
        rv = set()
        for e in self.unsigned_txs_out:
            rv.add(hash160_sec_to_bitcoin_address(ProxyTxOut.address_h160_from_script(e.script), addrversion=proxy.addrversion))
        return rv
