// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudLedger {
    struct TransactionRecord {
        string  transactionId;
        bool    isFraud;
        uint256 confidenceScore;   // stored as integer x100 (e.g. 87.5 → 8750)
        uint256 timestamp;
        address recordedBy;
    }

    mapping(string => TransactionRecord) public records;
    string[] public transactionIds;

    event TransactionRecorded(
        string  indexed transactionId,
        bool    isFraud,
        uint256 confidenceScore,
        uint256 timestamp
    );

    function recordTransaction(
        string  memory _txId,
        bool    _isFraud,
        uint256 _confidenceScore
    ) public {
        require(bytes(records[_txId].transactionId).length == 0, "Already recorded");

        records[_txId] = TransactionRecord({
            transactionId:   _txId,
            isFraud:         _isFraud,
            confidenceScore: _confidenceScore,
            timestamp:       block.timestamp,
            recordedBy:      msg.sender
        });
        transactionIds.push(_txId);

        emit TransactionRecorded(_txId, _isFraud, _confidenceScore, block.timestamp);
    }

    function getRecord(string memory _txId)
        public view
        returns (bool isFraud, uint256 confidenceScore, uint256 timestamp)
    {
        TransactionRecord memory r = records[_txId];
        return (r.isFraud, r.confidenceScore, r.timestamp);
    }

    function getTotalRecords() public view returns (uint256) {
        return transactionIds.length;
    }
}